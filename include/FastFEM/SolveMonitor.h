#pragma once

#include <atomic>

namespace fastfem {
struct SolveMonitor {
    SolveMonitor() : Progress(OwnedProgress), CancelRequested(OwnedCancelRequested) {}
    SolveMonitor(std::atomic<float> &progress, std::atomic<bool> &cancel_requested) : Progress(progress), CancelRequested(cancel_requested) {}

    void RequestCancel() { CancelRequested.store(true, std::memory_order_relaxed); }
    bool Cancelled() const { return CancelRequested.load(std::memory_order_relaxed); }

private:
    std::atomic<float> OwnedProgress{};
    std::atomic<bool> OwnedCancelRequested{};

public:
    std::atomic<float> &Progress;
    std::atomic<bool> &CancelRequested;
};
} // namespace fastfem
