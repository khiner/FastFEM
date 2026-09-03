#pragma once

#include <cstdlib>
#include <stdexcept>

inline void ConfigureAccelerateSparseExecution() {
    static const bool configured = [] {
#if FASTFEM_PARALLEL_SPARSE
        const int status = unsetenv("VECLIB_MAXIMUM_THREADS");
#else
        const int status = setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
#endif
        if (status != 0) throw std::runtime_error("Failed to configure Accelerate sparse execution.");
        return true;
    }();
    (void)configured;
}
