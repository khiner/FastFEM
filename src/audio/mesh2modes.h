#pragma once

#include "MassProperties.h"
#include "ModalEigenSummary.h"
#include "ModalModes.h"
#include <Eigen/Core>

#include <atomic>
#include <algorithm>
#include <memory>
#include <optional>
#include <span>

struct TetMesh;

namespace modal {
struct SolveMonitor {
    std::atomic<float> Progress{};
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

// Wall-clock seconds per solve stage, with problem-size counters.
// OpSolve is the shift-inverted linear solves, a subset of Iterate.
struct SolveProfile {
    double MassProps{}, QuadMesh{}, Assemble{}, SampleExcite{};
    double Factorize{}, Iterate{}, OpSolve{}, Extract{};
    double PhysicalResidual{}, MassOrthogonality{};
    uint32_t Dofs{}, StiffnessNonZeros{}, OpApplications{}, Restarts{};
    bool TopologyReuse{}, AssemblyReuse{}, SymbolicReuse{};

    SolveProfile &operator+=(const SolveProfile &o) {
        MassProps += o.MassProps;
        QuadMesh += o.QuadMesh;
        Assemble += o.Assemble;
        SampleExcite += o.SampleExcite;
        Factorize += o.Factorize;
        Iterate += o.Iterate;
        OpSolve += o.OpSolve;
        Extract += o.Extract;
        PhysicalResidual = std::max(PhysicalResidual, o.PhysicalResidual);
        MassOrthogonality = std::max(MassOrthogonality, o.MassOrthogonality);
        Dofs += o.Dofs;
        StiffnessNonZeros += o.StiffnessNonZeros;
        OpApplications += o.OpApplications;
        Restarts += o.Restarts;
        TopologyReuse = TopologyReuse || o.TopologyReuse;
        AssemblyReuse = AssemblyReuse || o.AssemblyReuse;
        SymbolicReuse = SymbolicReuse || o.SymbolicReuse;
        return *this;
    }
};

struct ModalResult {
    ModalModes Modes;
    MassProperties MassProps;
    SolveProfile Profile;
    ModalEigenSummary Summary; // Raw eigenpairs sampled at the excitation positions
    Eigen::MatrixXf Basis; // Full eigenvector basis, filled when SolveReuse::KeepBasis
    // The index into Modes.Positions of each requested excitation position, in request order.
    // Requests reaching the same tet point share one entry there.
    std::vector<uint32_t> SamplePointOfExcitation;
};

// mesh2modes uses a process-wide one-entry cache for topology and assembly.
// SolveCache preserves block-sparse numeric storage and symbolic factorization across sequential compatible solves.
// Pass an explicit cache to exchange persistent memory for lower repeated-solve latency.
struct SolveCache {
    struct State;

    std::unique_ptr<State> Reuse;

    SolveCache();
    ~SolveCache();
};

struct SolveReuse {
    const Eigen::MatrixXf *SeedBasis{};
    SolveCache *Cache{}; // Optional lifetime override for the bounded default cache
    bool KeepBasis{}; // Fill ModalResult::Basis
};

// FEM modal analysis over quadratic (10-node) tetrahedral elements.
// Tet geometry is in SI meters, so frequencies are in Hz and eigenvectors are mass-normalized.
// Each excitation position (SI) is sampled at its nearest tet point, and positions reaching the same point become one.
// The result therefore holds one position and one shape row per distinct point.
// `baked_scale` (the node's world scale) recovers node-local sample positions.
// `monitor` (optional) receives solve progress and is polled for cooperative cancellation
// between stages and eigensolver iterations. A cancelled solve returns an empty result.
ModalResult mesh2modes(const TetMesh &, const AcousticMaterialProperties &, const std::vector<vec3> &excite_positions, vec3 baked_scale, SolverConfig config = {}, SolveReuse reuse = {}, SolveMonitor *monitor = nullptr);

// Mode frequencies, T60s, and shapes from raw eigenpairs: filter to the audible window, apply
// damping and optional fundamental scaling. `shapes` holds each excitation position's mode-shape
// vector per eigenpair, scaled by `shape_scale` into the result.
ModalModes PostprocessModes(std::span<const double> eigenvalues, const std::vector<std::vector<vec3>> &shapes, float shape_scale, const AcousticMaterialProperties &, const SolverConfig &, std::vector<vec3> positions);

// Exact re-derivation of the modal model under a material edit at unchanged tet inputs: Young's
// modulus and density scale the FEM matrices linearly, so eigenvalues scale by (E'/E)/(rho'/rho)
// and mass-normalized shapes by 1/sqrt(rho'/rho). Positions and baked scale carry over
// from `current`. Empty when the edit is not exactly scalable (Poisson ratio differs).
std::optional<ModalModes> RescaleModes(const ModalEigenSummary &, const ModalModes &current, const AcousticMaterialProperties &, SolverConfig config = {});
} // namespace modal
