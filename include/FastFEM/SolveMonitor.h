#pragma once

#include <atomic>
#include <cstdint>

namespace fastfem {
enum struct SolveStage : uint8_t {
    PreparingSurface,
    GeneratingTetrahedra,
    BuildingFiniteCellGrid,
    ComputingMassProperties,
    BuildingTopology,
    AssemblingOperators,
    Factorizing,
    SolvingEigenproblem,
    SamplingModes,
    Finalizing,
    Complete,
};

struct SolveMonitor {
    std::atomic<float> Progress{};
    std::atomic<bool> CancelRequested{};
    std::atomic<SolveStage> Stage{SolveStage::PreparingSurface};
};
} // namespace fastfem
