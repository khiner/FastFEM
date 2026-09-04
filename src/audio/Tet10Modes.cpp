#include "Tet10Modes.h"

#include "AcousticMaterialProperties.h"
#include "GeneralizedEigenSolver.h"
#include "MassPropertiesAccumulator.h"
#include "ModalResultBuilder.h"
#include "Tet10Assembler.h"
#include "Tet10Cholesky.h"
#include "numeric/vec3.h"

#include "mesh/TetMesh.h"
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
    std::shared_ptr<const AssembledPencil> Assembly;
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

std::shared_ptr<const modal::AssembledPencil> AcquireAssembly(
    modal::SolveCache &cache, const modal::Tet10Assembler &fem, const AcousticMaterialProperties &material, bool &reused
) {
    const std::lock_guard lock{cache.Reuse->Mutex};
    if (cache.Reuse->Topology != fem.State) {
        reused = false;
        return std::make_shared<modal::AssembledPencil>(fem.AssembleLower());
    }
    const auto properties = modal::SolveCache::State::Elastic(material);
    if (cache.Reuse->Assembly && cache.Reuse->AssemblyMaterial == properties) {
        reused = true;
        return cache.Reuse->Assembly;
    }
    if (cache.Reuse->Assembly && cache.Reuse->AssemblyMaterial && cache.Reuse->AssemblyMaterial->PoissonRatio == properties.PoissonRatio) {
        auto scaled = std::make_shared<modal::AssembledPencil>(*cache.Reuse->Assembly);
        const double mass_scale = properties.Density / cache.Reuse->AssemblyMaterial->Density;
        const double stiffness_scale = properties.YoungModulus / cache.Reuse->AssemblyMaterial->YoungModulus;
        for (double &value : scaled->Mass.Values) value *= mass_scale;
        for (double &value : scaled->Stiffness.Values) value *= stiffness_scale;
        cache.Reuse->Assembly = std::move(scaled);
        reused = true;
    } else {
        cache.Reuse->Assembly = std::make_shared<modal::AssembledPencil>(fem.AssembleLower());
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
    const numeric::SparseMatrix &M,
    const numeric::SparseMatrix &K,
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
    const bool use_subspace = reuse.SeedBasis != nullptr && reuse.SeedBasis->rows() == n &&
        reuse.SeedBasis->cols() >= fem_n_modes;
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
            profile.PhysicalResidual = numeric::Maximum(subspace.RelativeResiduals.Last(fem_n_modes - first_physical));
        profile.MassOrthogonality = subspace.MassOrthogonalityError;
    }
    if (!subspace.Converged) return subspace;
    profile.Iterate = SecondsSince(eig_start) - profile.Factorize;
    return subspace;
}
} // namespace

modal::ModalResult modal::SolveTet10Modes(const TetMesh &input_tets, const AcousticMaterialProperties &material, const std::vector<vec3> &excite_positions, vec3 baked_scale, SolverConfig config, SolveReuse reuse, SolveMonitor *monitor) {
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
    profile.StiffnessNonZeros = uint32_t(K.NonZeros());
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
        for (size_t mode = 0; mode < eigenpairs.Eigenvalues.size(); ++mode)
            for (uint component = 0; component < 3; ++component)
                shapes[point][size_t(mode)][component] = eigenpairs.Eigenvectors(3 * excite_points[point] + component, mode);
    numeric::Matrix<float> basis;
    if (reuse.KeepBasis) basis = numeric::Cast<float>(eigenpairs.Eigenvectors.View());
    return BuildModalResult({eigenpairs.Eigenvalues.begin(), eigenpairs.Eigenvalues.end()}, std::move(shapes), material, config, std::move(positions), baked_scale, std::move(mass_props), profile, std::move(basis), std::move(sample_point_of));
}
