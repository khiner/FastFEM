#pragma once

#include <FastFEM/SolveMonitor.h>

namespace fastfem {
inline bool SolveCancelled(const SolveMonitor *monitor) {
    return monitor && monitor->CancelRequested.load(std::memory_order_relaxed);
}

inline void SetSolveProgress(SolveMonitor *monitor, float progress, SolveStage stage) {
    if (!monitor) return;
    monitor->Progress.store(progress, std::memory_order_relaxed);
    monitor->Stage.store(stage, std::memory_order_relaxed);
}
} // namespace fastfem
