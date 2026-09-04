#pragma once

#include "MassProperties.h"
#include "ModalEigenSummary.h"
#include "ModalModes.h"
#include "numeric/Matrix.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

struct TetMesh;

namespace modal {
// Shares progress and cancellation state with a running solve.
struct SolveMonitor {
    std::atomic<float> Progress{}; // Fraction complete, or zero while indeterminate
    std::atomic<bool> CancelRequested{};

    void RequestCancel() { CancelRequested.store(true, std::memory_order_relaxed); }
    bool Cancelled() const { return CancelRequested.load(std::memory_order_relaxed); }
};

// MinModeFreq also sets the eigensolver shift to -(2*pi*MinModeFreq)^2.
struct SolverConfig {
    float MinModeFreq{20}; // Hz
    float MaxModeFreq{16'000}; // Hz
    uint32_t NumModes{30}; // Synthesized modes kept from the FEM eigenpairs
    uint32_t NumFemModes{45}; // Eigenpairs requested from the eigensolver
    double Tolerance{1e-8}; // Eigensolver convergence tolerance
    uint32_t MaxRestarts{100}; // Eigensolver restart limit
    std::optional<float> FundamentalFreq{}; // Scale mode freqs so the lowest mode is at this fundamental
};

// Records wall-clock seconds, problem sizes, reuse, and certification diagnostics for one solve.
// OpSolve measures shift-inverted linear solves within Iterate.
struct SolveProfile {
    double MassProps{}, QuadMesh{}, Assemble{}, SampleExcite{};
    double Factorize{}, Iterate{}, OpSolve{}, Extract{};
    double PhysicalResidual{}, MassOrthogonality{};
    uint32_t Dofs{}, StiffnessNonZeros{}, OpApplications{}, Restarts{};
    bool TopologyReuse{}, AssemblyReuse{}, SymbolicReuse{};
};

struct ModalResult {
    ModalModes Modes;
    MassProperties MassProps;
    SolveProfile Profile;
    ModalEigenSummary Summary; // Raw eigenpairs sampled at the excitation positions
    numeric::Matrix<float> Basis; // Full eigenvector basis, filled when SolveReuse::KeepBasis
    // Maps each requested excitation position to Modes.Positions in request order.
    // Excitation positions mapped to one tetrahedral point share one entry.
    std::vector<uint32_t> SamplePointOfExcitation;
};

// SolveTet10Modes uses a process-wide one-entry cache for topology and assembly.
// SolveCache preserves block-sparse numeric storage and symbolic factorization across sequential compatible solves.
// Pass an explicit cache to exchange persistent memory for lower repeated-solve latency.
struct SolveCache {
    struct State;

    std::unique_ptr<State> Reuse;

    SolveCache();
    ~SolveCache();
};

struct SolveReuse {
    const numeric::Matrix<float> *SeedBasis{};
    SolveCache *Cache{}; // Optional lifetime override for the bounded default cache
    bool KeepBasis{}; // Fill ModalResult::Basis
};

// Returns mass-normalized modes from quadratic tetrahedra whose coordinates and excitation positions use SI meters.
// The result contains one position and shape row per distinct nearest tetrahedral point.
// `baked_scale` converts the sampled positions to node-local coordinates.
// `monitor` receives progress and supports cancellation between stages and eigensolver iterations.
// Cancellation returns an empty result.
ModalResult SolveTet10Modes(const TetMesh &, const AcousticMaterialProperties &, const std::vector<vec3> &excite_positions, vec3 baked_scale, SolverConfig config = {}, SolveReuse reuse = {}, SolveMonitor *monitor = nullptr);

// Returns exact rescaled modes for unchanged tetrahedral inputs and unchanged Poisson ratio.
// Young's modulus and density scale eigenvalues by (E'/E)/(rho'/rho) and mass-normalized shapes by 1/sqrt(rho'/rho).
// The result copies positions and baked scale from `current`.
// A Poisson-ratio change returns nullopt.
std::optional<ModalModes> RescaleModes(const ModalEigenSummary &, const ModalModes &current, const AcousticMaterialProperties &, SolverConfig config = {});

} // namespace modal
