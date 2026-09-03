// Eigen delegates large dense products used by Lanczos orthogonalization to Accelerate BLAS.
#define EIGEN_USE_BLAS

#include "mesh2modes.h"

#include "AcousticMaterialProperties.h"
#include "BlockSparseCholesky.h"
#include "GeneralizedEigenSolver.h"
#include "Tet10Assembler.h"
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
    std::mutex BlockFactorMutex;
    std::unique_ptr<BlockSparseCholesky> BlockFactor;
    std::shared_ptr<const Tet10Assembler::Topology> BlockTopology;
    std::optional<ElasticProperties> BlockMaterial;
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

quat QuaternionFromRotation(const Eigen::Matrix3f &rotation) {
    const auto m = [&](int column, int row) { return rotation(row, column); };
    const std::array candidates{
        m(0, 0) + m(1, 1) + m(2, 2),
        m(0, 0) - m(1, 1) - m(2, 2),
        m(1, 1) - m(0, 0) - m(2, 2),
        m(2, 2) - m(0, 0) - m(1, 1),
    };
    const auto biggest = std::ranges::max_element(candidates) - candidates.begin();
    const float value = std::sqrt(candidates[biggest] + 1.f) * .5f;
    const float scale = .25f / value;
    quat result;
    switch (biggest) {
        case 0: result = {value, (m(1, 2) - m(2, 1)) * scale, (m(2, 0) - m(0, 2)) * scale, (m(0, 1) - m(1, 0)) * scale}; break;
        case 1: result = {(m(1, 2) - m(2, 1)) * scale, value, (m(0, 1) + m(1, 0)) * scale, (m(2, 0) + m(0, 2)) * scale}; break;
        case 2: result = {(m(2, 0) - m(0, 2)) * scale, (m(0, 1) + m(1, 0)) * scale, value, (m(1, 2) + m(2, 1)) * scale}; break;
        case 3: result = {(m(0, 1) - m(1, 0)) * scale, (m(2, 0) + m(0, 2)) * scale, (m(1, 2) + m(2, 1)) * scale, value}; break;
    }
    const float norm = std::sqrt(result.w * result.w + result.x * result.x + result.y * result.y + result.z * result.z);
    return {result.w / norm, result.x / norm, result.y / norm, result.z / norm};
}

// Assigns one quarter of each tetrahedron's volume to each vertex for rigid-body mass properties.
// `scale` maps tet coordinates to node-local space; `length_to_si` maps lengths to meters.
MassProperties ComputeMassProperties(const TetMesh &tets, double density, vec3 scale, double length_to_si) {
    const size_t nverts = tets.Points.size();
    const dvec3 inv_scale{1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z};
    std::vector<dvec3> pos(nverts);
    for (size_t i = 0; i < nverts; ++i) pos[i] = tets.Points[i] * inv_scale;

    std::vector<double> vol(nverts, 0.0);
    for (const auto &t : tets.Tets) {
        const double quarter = GetTetVolume(pos[t[0]], pos[t[1]], pos[t[2]], pos[t[3]]) * 0.25;
        for (int c = 0; c < 4; ++c) vol[t[c]] += quarter;
    }

    double total = 0;
    dvec3 com{0};
    for (size_t i = 0; i < nverts; ++i) {
        total += vol[i];
        com += vol[i] * pos[i];
    }
    if (total <= 0) return {};
    com /= total;

    // Point-mass inertia sums vol*(|r|^2*I - r*r^T) about the center of mass.
    // Conversion to SI scales this volume-length-squared integral by length_to_si^5.
    const double s = length_to_si;
    Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < nverts; ++i) {
        const dvec3 r = pos[i] - com;
        const double rr = numeric::Dot(r, r);
        inertia(0, 0) += vol[i] * (rr - r.x * r.x);
        inertia(1, 1) += vol[i] * (rr - r.y * r.y);
        inertia(2, 2) += vol[i] * (rr - r.z * r.z);
        inertia(0, 1) -= vol[i] * r.x * r.y;
        inertia(0, 2) -= vol[i] * r.x * r.z;
        inertia(1, 2) -= vol[i] * r.y * r.z;
    }
    inertia(1, 0) = inertia(0, 1);
    inertia(2, 0) = inertia(0, 2);
    inertia(2, 1) = inertia(1, 2);
    inertia *= density * s * s * s * s * s;

    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(inertia);
    const auto &evals = es.eigenvalues();
    const auto &evecs = es.eigenvectors();
    Eigen::Matrix3f axes = evecs.cast<float>();
    if (axes.determinant() < 0) axes.col(0) = -axes.col(0);

    return {
        density * total * s * s * s,
        vec3{com},
        vec3{float(evals[0]), float(evals[1]), float(evals[2])},
        QuaternionFromRotation(axes),
    };
}

struct ComputeModesOpts {
    modal::SolverConfig Config{};
    std::vector<uint> ExPos{}; // Excitation positions
    std::vector<vec3> Positions{}; // Node-local positions of each excitation position, parallel to ExPos
    AcousticMaterialProperties Material{};
    modal::SolveReuse Reuse{};
    modal::SolveMonitor *Monitor{};
};

struct Tet10BlockShiftInvert {
    using Scalar = double;
    using Index = Eigen::Index;

    const modal::Tet10Assembler &Fem;
    const AcousticMaterialProperties &Material;
    double &FactorizeSeconds, &SolveSeconds;
    bool &SymbolicReuse;
    modal::SolveCache::State *Cache{};
    std::unique_lock<std::mutex> CacheLock;
    std::unique_ptr<BlockSparseCholesky> LocalFactor;
    BlockSparseCholesky *Factor{};

    Tet10BlockShiftInvert(
        const modal::Tet10Assembler &fem, const AcousticMaterialProperties &material, double &factorize_seconds,
        double &solve_seconds, bool &symbolic_reuse, modal::SolveCache *cache
    ) : Fem(fem), Material(material), FactorizeSeconds(factorize_seconds), SolveSeconds(solve_seconds),
        SymbolicReuse(symbolic_reuse), Cache(cache ? cache->Reuse.get() : nullptr),
        CacheLock(Cache ? std::unique_lock{Cache->BlockFactorMutex, std::try_to_lock} : std::unique_lock<std::mutex>{}) {}

    Index rows() const { return Fem.Dofs(); }
    Index cols() const { return Fem.Dofs(); }

    void set_shift(const Scalar &sigma) {
        const auto start = std::chrono::steady_clock::now();
        if (!Factor) {
            bool reused{};
            if (CacheLock.owns_lock()) {
                const auto properties = modal::SolveCache::State::Elastic(Material);
                reused = bool(Cache->BlockFactor);
                if (!Cache->BlockFactor) {
                    Cache->BlockFactor = std::make_unique<BlockSparseCholesky>(Fem);
                } else if (Cache->BlockTopology == Fem.State && Cache->BlockMaterial && Cache->BlockMaterial->PoissonRatio == properties.PoissonRatio) {
                    if (*Cache->BlockMaterial != properties)
                        Cache->BlockFactor->ScalePencil(properties.YoungModulus / Cache->BlockMaterial->YoungModulus, properties.Density / Cache->BlockMaterial->Density);
                } else {
                    try {
                        Cache->BlockFactor->Reassemble(Fem);
                    } catch (const std::invalid_argument &) {
                        Cache->BlockFactor = std::make_unique<BlockSparseCholesky>(Fem);
                        reused = false;
                    }
                }
                Cache->BlockTopology = Fem.State;
                Cache->BlockMaterial = properties;
                Factor = Cache->BlockFactor.get();
            } else {
                LocalFactor = std::make_unique<BlockSparseCholesky>(Fem);
                Factor = LocalFactor.get();
            }
            SymbolicReuse = reused;
        }
        Factor->SetShift(sigma);
        FactorizeSeconds += SecondsSince(start);
    }

    void perform_op(const Scalar *input, Scalar *output) const {
        const auto start = std::chrono::steady_clock::now();
        Factor->Solve(input, output);
        SolveSeconds += SecondsSince(start);
    }

    void solve_panel(const Scalar *input, Scalar *output, int width) const {
        const auto start = std::chrono::steady_clock::now();
        Factor->Solve(input, output, width);
        SolveSeconds += SecondsSince(start);
    }
};

ModalModes ComputeModes(
    const modal::Tet10Assembler &fem,
    const Eigen::SparseMatrix<double> &M,
    const Eigen::SparseMatrix<double> &K,
    uint num_vertices, uint vertex_dim,
    ComputeModesOpts opts,
    modal::SolveProfile &profile,
    ModalEigenSummary &summary_out, Eigen::MatrixXf *basis_out
) {
    using OpType = Tet10BlockShiftInvert;
    const auto &config = opts.Config;
    const uint n = num_vertices * vertex_dim;
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
    const auto subspace = modal::detail::SolveGeneralizedEigenproblem(
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
    if (!subspace.Converged) return {};
    profile.Iterate = SecondsSince(eig_start) - profile.Factorize;
    const Eigen::VectorXd &eigenvalues = subspace.Eigenvalues;
    const Eigen::MatrixXd &eigenvectors = subspace.Eigenvectors;
    std::vector<std::vector<vec3>> shapes(opts.ExPos.size(), std::vector<vec3>(fem_n_modes));
    for (size_t ex_pos = 0; ex_pos < shapes.size(); ++ex_pos) {
        const uint ev_i = vertex_dim * opts.ExPos[ex_pos];
        for (uint mode = 0; mode < fem_n_modes; ++mode) {
            for (uint vi = 0; vi < vertex_dim; ++vi) shapes[ex_pos][mode][vi] = eigenvectors(ev_i + vi, mode);
        }
    }
    summary_out.Eigenvalues = {eigenvalues.begin(), eigenvalues.end()};
    summary_out.Shapes = shapes;
    summary_out.SolvedMaterial = opts.Material;
    if (basis_out) {
        const Eigen::MatrixXd mass_vectors = M.selfadjointView<Eigen::Lower>() * eigenvectors;
        const Eigen::MatrixXd stiffness_vectors = K.selfadjointView<Eigen::Lower>() * eigenvectors;
        const Eigen::MatrixXd residual = stiffness_vectors - mass_vectors * eigenvalues.asDiagonal();
        for (uint32_t mode = 0; mode < fem_n_modes; ++mode) {
            if (std::abs(eigenvalues[mode]) <= -sigma * 1e-4) continue;
            const double scale = stiffness_vectors.col(mode).norm() + std::abs(eigenvalues[mode]) * mass_vectors.col(mode).norm();
            profile.PhysicalResidual = std::max(profile.PhysicalResidual, residual.col(mode).norm() / scale);
        }
        profile.MassOrthogonality = (eigenvectors.transpose() * mass_vectors - Eigen::MatrixXd::Identity(fem_n_modes, fem_n_modes)).norm();
        *basis_out = eigenvectors.cast<float>();
    }

    return modal::PostprocessModes(summary_out.Eigenvalues, shapes, 1.f, opts.Material, config, std::move(opts.Positions));
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

modal::ModalResult modal::mesh2modes(const TetMesh &input_tets, const AcousticMaterialProperties &material, const std::vector<vec3> &excite_positions, vec3 baked_scale, SolverConfig config, SolveReuse reuse, SolveMonitor *monitor) {
    const TetMesh tets = FilterDegenerate(input_tets);
    SolveProfile profile;
    auto &cache = reuse.Cache ? *reuse.Cache : DefaultSolveCache();
    const double length_to_si = (double(baked_scale.x) + baked_scale.y + baked_scale.z) / 3.0;
    auto mass_props = Timed(profile.MassProps, [&] { return ComputeMassProperties(tets, material.Density, baked_scale, length_to_si); });

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
    ModalEigenSummary summary;
    Eigen::MatrixXf basis;
    auto modes = ComputeModes(fem, M, K, fem.NumNodes, 3, {
                                                              .Config = std::move(config),
                                                              .ExPos = std::move(excite_points),
                                                              .Positions = std::move(positions),
                                                              .Material = material,
                                                              .Reuse = reuse,
                                                              .Monitor = monitor,
                                                          },
                              profile, summary, reuse.KeepBasis ? &basis : nullptr);
    return {std::move(modes), std::move(mass_props), profile, std::move(summary), std::move(basis), std::move(sample_point_of)};
}
