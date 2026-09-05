#include "SparseExecution.h"

#include <cstdlib>
#include <stdexcept>

// Accelerate may cache the thread limit before the first sparse operation.
[[gnu::constructor]] void ConfigureAccelerateSparseExecution() {
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
