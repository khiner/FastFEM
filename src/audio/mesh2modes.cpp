// Eigen delegates large dense products in block eigensolver orthogonalization to Accelerate BLAS.
#define EIGEN_USE_BLAS

#include "mesh2modes.h"

#include "AcousticMaterialProperties.h"
#include "GeneralizedEigenSolver.h"
#include "MassPropertiesAccumulator.h"
#include "ModalResultBuilder.h"
#include "Tet10Assembler.h"
#include "Tet10Cholesky.h"
#include "numeric/vec3.h"

#include "mesh/TetMesh.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <mutex>
#include <numbers>
#include <optional>
#include <unordered_map>

using uint = uint32_t;

struct modal::SolveCache::State {
    struct ElasticProperties {
        double Density, YoungModulus, PoissonRatio;
        auto operator<=>(const ElasticProperties &) const = default;
    };

    static ElasticProperties Elastic(const AcousticMaterialProperties &material) {
        return {material.Density, material.YoungModulus, material.PoissonRatio};
    }

    std::mutex Mutex;
    TetMesh Mesh;
    std::shared_ptr<const Tet10Assembler::Topology> Topology;
    std::shared_ptr<const Tet10Assembler::AssembledLower> Assembly;
    std::optional<ElasticProperties> AssemblyMaterial;
    std::mutex Tet10FactorMutex;
    std::unique_ptr<Tet10Cholesky> Tet10Factor;
    std::shared_ptr<const Tet10Assembler::Topology> Tet10FactorTopology;
    std::optional<ElasticProperties> Tet10FactorMaterial;
};

modal::SolveCache::SolveCache() : Reuse(std::make_unique<State>()) {}
modal::SolveCache::~SolveCache() = default;

namespace {
constexpr uint WarmOversampling{15};
constexpr double WarmTolerance{1e-4};
constexpr uint WarmRefinementIterations{10};

modal::SolveCache &DefaultSolveCache() {
    static modal::SolveCache cache;
    return cache;
}

std::shared_ptr<const modal::Tet10Assembler::Topology> AcquireTopology(
    modal::SolveCache &cache, const TetMesh &mesh, bool &reused
) {
    const std::lock_guard lock{cache.Reuse->Mutex};
    const bool same = cache.Reuse->Topology && cache.Reuse->Mesh.Points == mesh.Points && cache.Reuse->Mesh.Tets == mesh.Tets;
    if (!same) {
        cache.Reuse->Mesh = mesh;
        cache.Reuse->Topology = modal::Tet10Assembler::BuildTopology(mesh);
        cache.Reuse->Assembly.reset();
        cache.Reuse->AssemblyMaterial.reset();
    }
    reused = same;
    return cache.Reuse->Topology;
}

std::shared_ptr<const modal::Tet10Assembler::AssembledLower> AcquireAssembly(
    modal::SolveCache &cache, const modal::Tet10Assembler &fem, const AcousticMaterialProperties &material, bool &reused
) {
    const std::lock_guard lock{cache.Reuse->Mutex};
    if (cache.Reuse->Topology != fem.State) {
        reused = false;
        return std::make_shared<modal::Tet10Assembler::AssembledLower>(fem.AssembleLower());
    }
    const auto properties = modal::SolveCache::State::Elastic(material);
    if (cache.Reuse->Assembly && cache.Reuse->AssemblyMaterial == properties) {
        reused = true;
        return cache.Reuse->Assembly;
    }
    if (cache.Reuse->Assembly && cache.Reuse->AssemblyMaterial && cache.Reuse->AssemblyMaterial->PoissonRatio == properties.PoissonRatio) {
        auto scaled = std::make_shared<modal::Tet10Assembler::AssembledLower>(*cache.Reuse->Assembly);
        scaled->Mass *= properties.Density / cache.Reuse->AssemblyMaterial->Density;
        scaled->Stiffness *= properties.YoungModulus / cache.Reuse->AssemblyMaterial->YoungModulus;
        cache.Reuse->Assembly = std::move(scaled);
        reused = true;
    } else {
        cache.Reuse->Assembly = std::make_shared<modal::Tet10Assembler::AssembledLower>(fem.AssembleLower());
        reused = false;
    }
    cache.Reuse->AssemblyMaterial = properties;
    return cache.Reuse->Assembly;
}

double SecondsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

auto Timed(double &seconds, auto &&compute) {
    const auto start = std::chrono::steady_clock::now();
    auto result = compute();
    seconds = SecondsSince(start);
    return result;
}

// Degenerate elements have zero physical contribution.
// Removing them prevents inverse-determinant basis gradients from producing non-finite stiffness entries.
TetMesh FilterDegenerate(const TetMesh &tets) {
    TetMesh clean;
    clean.Points = tets.Points;
    clean.Tets.reserve(tets.Tets.size());
    for (const auto &t : tets.Tets) {
        const dvec3 &a = tets.Points[t[0]];
        const dvec3 r0 = tets.Points[t[1]] - a, r1 = tets.Points[t[2]] - a, r2 = tets.Points[t[3]] - a;
        const double det = std::abs(numeric::Dot(r0, numeric::Cross(r1, r2)));
        double lmax_sq = 0;
        for (uint i = 0; i < 4; ++i) {
            for (uint j = i + 1; j < 4; ++j) {
                const dvec3 d = tets.Points[t[i]] - tets.Points[t[j]];
                lmax_sq = std::max(lmax_sq, numeric::Dot(d, d));
            }
        }
        if (det > 1e-12 * lmax_sq * std::sqrt(lmax_sq)) clean.Tets.push_back(t);
    }
    return clean;
}

double GetTetDeterminant(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d) {
    return numeric::Dot(d - a, numeric::Cross(b - a, c - a));
}
double GetTetVolume(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d) {
    return std::abs(GetTetDeterminant(a, b, c, d)) / 6.0;
}

// Assigns one quarter of each tetrahedron's volume to each vertex for rigid-body mass properties.
// `scale` maps tet coordinates to node-local space; `length_to_si` maps lengths to meters.
MassProperties ComputeTetMassProperties(const TetMesh &tets, double density, vec3 scale, double length_to_si) {
    const size_t nverts = tets.Points.size();
    const dvec3 inv_scale{1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z};
    std::vector<dvec3> pos(nverts);
    for (size_t i = 0; i < nverts; ++i) pos[i] = tets.Points[i] * inv_scale;

    std::vector<double> vol(nverts, 0.0);
    for (const auto &t : tets.Tets) {
        const double quarter = GetTetVolume(pos[t[0]], pos[t[1]], pos[t[2]], pos[t[3]]) * 0.25;
        for (int c = 0; c < 4; ++c) vol[t[c]] += quarter;
    }

    return modal::ComputeMassProperties(pos, vol, density, length_to_si);
}

struct Tet10SolveOptions {
    modal::SolverConfig Config{};
    AcousticMaterialProperties Material{};
    modal::SolveReuse Reuse{};
    modal::SolveMonitor *Monitor{};
};

struct Tet10ShiftInvert {
    const modal::Tet10Assembler &Fem;
    const AcousticMaterialProperties &Material;
    double &FactorizeSeconds, &SolveSeconds;
    bool &SymbolicReuse;
    modal::SolveCache::State *Cache{};
    std::unique_lock<std::mutex> CacheLock;
    std::unique_ptr<modal::Tet10Cholesky> LocalFactor;
    modal::Tet10Cholesky *Factor{};

    Tet10ShiftInvert(
        const modal::Tet10Assembler &fem, const AcousticMaterialProperties &material, double &factorize_seconds,
        double &solve_seconds, bool &symbolic_reuse, modal::SolveCache *cache
    ) : Fem(fem), Material(material), FactorizeSeconds(factorize_seconds), SolveSeconds(solve_seconds),
        SymbolicReuse(symbolic_reuse), Cache(cache ? cache->Reuse.get() : nullptr),
        CacheLock(Cache ? std::unique_lock{Cache->Tet10FactorMutex, std::try_to_lock} : std::unique_lock<std::mutex>{}) {}

    void set_shift(double sigma) {
        const auto start = std::chrono::steady_clock::now();
        if (!Factor) {
            bool reused{};
            if (CacheLock.owns_lock()) {
                const auto properties = modal::SolveCache::State::Elastic(Material);
                reused = bool(Cache->Tet10Factor);
                if (!Cache->Tet10Factor) {
                    Cache->Tet10Factor = std::make_unique<modal::Tet10Cholesky>(Fem);
                } else if (Cache->Tet10FactorTopology == Fem.State && Cache->Tet10FactorMaterial && Cache->Tet10FactorMaterial->PoissonRatio == properties.PoissonRatio) {
                    if (*Cache->Tet10FactorMaterial != properties)
                        Cache->Tet10Factor->ScalePencil(properties.YoungModulus / Cache->Tet10FactorMaterial->YoungModulus, properties.Density / Cache->Tet10FactorMaterial->Density);
                } else {
                    try {
                        Cache->Tet10Factor->Reassemble(Fem);
                    } catch (const std::invalid_argument &) {
                        Cache->Tet10Factor = std::make_unique<modal::Tet10Cholesky>(Fem);
                        reused = false;
                    }
                }
                Cache->Tet10FactorTopology = Fem.State;
                Cache->Tet10FactorMaterial = properties;
                Factor = Cache->Tet10Factor.get();
            } else {
                LocalFactor = std::make_unique<modal::Tet10Cholesky>(Fem);
                Factor = LocalFactor.get();
            }
            SymbolicReuse = reused;
        }
        Factor->SetShift(sigma);
        FactorizeSeconds += SecondsSince(start);
    }

    void solve_panel(const double *input, double *output, int width) const {
        const auto start = std::chrono::steady_clock::now();
        Factor->Solve(input, output, width);
        SolveSeconds += SecondsSince(start);
    }
};

modal::eigensolver::GeneralizedEigenResult SolveTet10Eigenpairs(
    const modal::Tet10Assembler &fem,
    const Eigen::SparseMatrix<double> &M,
    const Eigen::SparseMatrix<double> &K,
    Tet10SolveOptions opts,
    modal::SolveProfile &profile
) {
    using OpType = Tet10ShiftInvert;
    const auto &config = opts.Config;
    const uint n = fem.Dofs();
    const uint fem_n_modes = std::min(config.NumFemModes, n - 1);
    const uint basis_size = std::min(std::max(fem_n_modes + (n >= 100000 ? 30u : 20u), 20u), n);
    // A negative shift makes K - sigma*M positive definite and places the smallest eigenvalues nearest the shift.
    const double shift_omega = 2 * std::numbers::pi * config.MinModeFreq;
    const double sigma = -shift_omega * shift_omega;
    auto *monitor = opts.Monitor;
    if (monitor && monitor->Cancelled()) return {};
    const auto &reuse = opts.Reuse;
    OpType op{fem, opts.Material, profile.Factorize, profile.OpSolve, profile.SymbolicReuse, reuse.Cache};
    if (monitor) monitor->Progress.store(0.3f, std::memory_order_relaxed);
    const bool use_subspace = reuse.SeedBasis != nullptr && reuse.SeedBasis->rows() == Eigen::Index(n) &&
        reuse.SeedBasis->cols() >= Eigen::Index(fem_n_modes);
    const auto eig_start = std::chrono::steady_clock::now();
    const auto subspace = modal::eigensolver::SolveGeneralizedEigenproblem(
        op, M, K,
        {
            .Count = fem_n_modes,
            .SubspaceSize = use_subspace ? std::min(fem_n_modes + WarmOversampling, n) : basis_size,
            .Shift = sigma,
            .IterationTolerance = use_subspace ? WarmTolerance : 1e-3,
            .ResidualTolerance = config.Tolerance,
            .MaxIterations = config.MaxRestarts,
            .MaxRefinementIterations = use_subspace ? WarmRefinementIterations : 50,
            .KrylovBlockWidth = 8,
            .KrylovSize = std::min(fem_n_modes + 115, n),
            .ExtendedKrylovSize = std::min(fem_n_modes + 203, n),
        },
        use_subspace ? reuse.SeedBasis : nullptr,
        {
            .Cancel = monitor ? &monitor->CancelRequested : nullptr,
            .Progress = monitor ? &monitor->Progress : nullptr,
            .ProgressBegin = 0.3f,
            .ProgressEnd = 0.95f,
        }
    );
    profile.OpApplications = subspace.OpApplications;
    profile.Restarts = subspace.Iterations;
    if (subspace.RelativeResiduals.size()) {
        const uint32_t first_physical = std::min(6u, fem_n_modes);
        if (first_physical < fem_n_modes)
            profile.PhysicalResidual = subspace.RelativeResiduals.tail(fem_n_modes - first_physical).maxCoeff();
        profile.MassOrthogonality = subspace.MassOrthogonalityError;
    }
    if (!subspace.Converged) return subspace;
    profile.Iterate = SecondsSince(eig_start) - profile.Factorize;
    return subspace;
}
} // namespace

ModalModes modal::PostprocessModes(std::span<const double> eigenvalues, const std::vector<std::vector<vec3>> &shapes, float shape_scale, const AcousticMaterialProperties &material, const SolverConfig &config, std::vector<vec3> positions) {
    const uint fem_n_modes = eigenvalues.size();
    std::vector<float> mode_freqs(fem_n_modes), mode_t60s(fem_n_modes);
    std::vector<double> omega_undamped(fem_n_modes);
    // The eigensolver shift scales the near-zero eigenvalue cutoff.
    const double shift_omega = 2 * std::numbers::pi * config.MinModeFreq;
    const double lambda_eps = shift_omega * shift_omega * 1e-10;
    for (uint mode = 0; mode < fem_n_modes; ++mode) {
        const double lambda_i = eigenvalues[mode];
        omega_undamped[mode] = lambda_i > lambda_eps ? std::sqrt(lambda_i) : 0;
    }

    const auto c_from_omega = [&material](double omega) { return material.Alpha + material.Beta * (omega * omega); };
    const auto damped_hz = [&](double omega, double c) {
        const double omega_d_sq = omega * omega - 0.25 * c * c;
        return omega_d_sq > 0 ? std::sqrt(omega_d_sq) / (2 * std::numbers::pi) : 0;
    };

    uint lowest_mode_i = fem_n_modes;
    float lowest_mode_freq_orig{0};
    for (uint mode = 0; mode < fem_n_modes; ++mode) {
        const double omega_i = omega_undamped[mode];
        if (omega_i <= 0) {
            mode_freqs[mode] = mode_t60s[mode] = 0.f;
            continue;
        }
        mode_freqs[mode] = damped_hz(omega_i, c_from_omega(omega_i));
        // The audible-frequency floor excludes numerically nonzero rigid-body modes.
        if (lowest_mode_i == fem_n_modes && mode_freqs[mode] >= config.MinModeFreq) {
            lowest_mode_i = mode;
            lowest_mode_freq_orig = mode_freqs[mode];
        }
    }
    if (lowest_mode_i == fem_n_modes) return {};

    static const double ln_1000 = std::log(1000);
    const float freq_scale = config.FundamentalFreq ? *config.FundamentalFreq / lowest_mode_freq_orig : 1.f;
    for (uint mode = lowest_mode_i; mode < fem_n_modes; ++mode) {
        const double omega_s = omega_undamped[mode] * freq_scale; // scaled rad/s
        const double c = c_from_omega(omega_s);
        mode_freqs[mode] = damped_hz(omega_s, c);
        mode_t60s[mode] = c > 0 ? (2 * ln_1000) / c : 0;
    }
    // Applying the upper frequency cutoff before fundamental scaling preserves higher modes after retuning.
    const float max_mode_freq = config.MaxModeFreq * std::max(1.f, freq_scale);
    uint highest_mode_i = fem_n_modes;
    while (highest_mode_i > lowest_mode_i && mode_freqs[highest_mode_i - 1] > max_mode_freq) --highest_mode_i;

    const uint n_modes = std::min({config.NumModes, fem_n_modes, highest_mode_i - lowest_mode_i});
    mode_freqs.erase(mode_freqs.begin(), mode_freqs.begin() + lowest_mode_i);
    mode_freqs.resize(n_modes);
    mode_t60s.erase(mode_t60s.begin(), mode_t60s.begin() + lowest_mode_i);
    mode_t60s.resize(n_modes);

    std::vector<std::vector<vec3>> out_shapes(shapes.size(), std::vector<vec3>(n_modes));
    for (size_t ex_pos = 0; ex_pos < shapes.size(); ++ex_pos) {
        for (uint mode = 0; mode < n_modes; ++mode) out_shapes[ex_pos][mode] = shapes[ex_pos][mode + lowest_mode_i] * shape_scale;
    }

    return {
        .Freqs = std::move(mode_freqs),
        .T60s = std::move(mode_t60s),
        .Shapes = std::move(out_shapes),
        .Positions = std::move(positions),
        .OriginalFundamentalFreq = lowest_mode_freq_orig,
    };
}

std::optional<ModalModes> modal::RescaleModes(const ModalEigenSummary &summary, const ModalModes &current, const AcousticMaterialProperties &material, SolverConfig config) {
    if (summary.Eigenvalues.empty() || material.PoissonRatio != summary.SolvedMaterial.PoissonRatio) return {};

    const double rho_ratio = material.Density / summary.SolvedMaterial.Density;
    const double eigenvalue_scale = (material.YoungModulus / summary.SolvedMaterial.YoungModulus) / rho_ratio;
    auto eigenvalues = summary.Eigenvalues;
    for (auto &v : eigenvalues) v *= eigenvalue_scale;

    auto modes = PostprocessModes(eigenvalues, summary.Shapes, float(1 / std::sqrt(rho_ratio)), material, config, current.Positions);
    modes.BakedScale = current.BakedScale;
    return modes;
}

modal::ModalResult modal::BuildModalResult(std::vector<double> eigenvalues, std::vector<std::vector<vec3>> shapes, const AcousticMaterialProperties &material, const SolverConfig &config, std::vector<vec3> positions, vec3 baked_scale, MassProperties mass_properties, SolveProfile profile, Eigen::MatrixXf basis, std::vector<uint32_t> sample_point_of) {
    ModalEigenSummary summary{
        .Eigenvalues = std::move(eigenvalues),
        .Shapes = std::move(shapes),
        .SolvedMaterial = material,
    };
    auto modes = PostprocessModes(summary.Eigenvalues, summary.Shapes, 1, material, config, std::move(positions));
    modes.BakedScale = baked_scale;
    return {std::move(modes), std::move(mass_properties), profile, std::move(summary), std::move(basis), std::move(sample_point_of)};
}

modal::ModalResult modal::mesh2modes(const TetMesh &input_tets, const AcousticMaterialProperties &material, const std::vector<vec3> &excite_positions, vec3 baked_scale, SolverConfig config, SolveReuse reuse, SolveMonitor *monitor) {
    const TetMesh tets = FilterDegenerate(input_tets);
    SolveProfile profile;
    auto &cache = reuse.Cache ? *reuse.Cache : DefaultSolveCache();
    const double length_to_si = (double(baked_scale.x) + baked_scale.y + baked_scale.z) / 3.0;
    auto mass_props = Timed(profile.MassProps, [&] { return ComputeTetMassProperties(tets, material.Density, baked_scale, length_to_si); });

    if (monitor) monitor->Progress.store(0.1f, std::memory_order_relaxed);
    const auto topology = Timed(profile.QuadMesh, [&] { return AcquireTopology(cache, tets, profile.TopologyReuse); });
    const Tet10Assembler fem{topology, material};
    const auto assembly = Timed(profile.Assemble, [&] { return AcquireAssembly(cache, fem, material, profile.AssemblyReuse); });
    const auto &M = assembly->Mass, &K = assembly->Stiffness;
    profile.Dofs = fem.Dofs();
    profile.StiffnessNonZeros = K.nonZeros();
    if (monitor && monitor->Cancelled()) return {};

    // Sample each excitation position at its nearest tetrahedral point and convert the result to node-local coordinates.
    // Excitation positions mapped to one tetrahedral point share one shape vector and sample point.
    auto [excite_points, positions, sample_point_of] = Timed(profile.SampleExcite, [&] {
        const dvec3 inv_scale{1.0 / baked_scale.x, 1.0 / baked_scale.y, 1.0 / baked_scale.z};
        std::vector<uint> points;
        std::vector<vec3> local;
        std::vector<uint32_t> remap(excite_positions.size());
        std::unordered_map<uint, uint32_t> sample_point_at;
        for (size_t i = 0; i < excite_positions.size(); ++i) {
            const dvec3 p{excite_positions[i]};
            double best = std::numeric_limits<double>::max();
            uint nearest = 0;
            for (uint v = 0; v < uint(tets.Points.size()); ++v) {
                const auto &q = tets.Points[v];
                if (const double d = numeric::Distance2(p, q); d < best) {
                    best = d;
                    nearest = v;
                }
            }
            const auto [entry, first] = sample_point_at.emplace(nearest, uint32_t(points.size()));
            if (first) {
                points.push_back(nearest);
                local.emplace_back(tets.Points[nearest] * inv_scale);
            }
            remap[i] = entry->second;
        }
        return std::tuple{std::move(points), std::move(local), std::move(remap)};
    });
    const auto eigenpairs = SolveTet10Eigenpairs(
        fem, M, K,
        {
            .Config = config,
            .Material = material,
            .Reuse = reuse,
            .Monitor = monitor,
        },
        profile
    );
    if (!eigenpairs.Converged) return {.MassProps = std::move(mass_props), .Profile = profile};

    std::vector<std::vector<vec3>> shapes(excite_points.size(), std::vector<vec3>(eigenpairs.Eigenvalues.size()));
    for (size_t point = 0; point < excite_points.size(); ++point)
        for (Eigen::Index mode = 0; mode < eigenpairs.Eigenvalues.size(); ++mode)
            for (uint component = 0; component < 3; ++component)
                shapes[point][size_t(mode)][component] = eigenpairs.Eigenvectors(3 * excite_points[point] + component, mode);
    Eigen::MatrixXf basis;
    if (reuse.KeepBasis) basis = eigenpairs.Eigenvectors.cast<float>();
    return BuildModalResult({eigenpairs.Eigenvalues.begin(), eigenpairs.Eigenvalues.end()}, std::move(shapes), material, config, std::move(positions), baked_scale, std::move(mass_props), profile, std::move(basis), std::move(sample_point_of));
}
