#include "SparseCholesky.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {
void ConfigureSparseExecution() {
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

struct BlockSparseLower {
    std::vector<long> ColumnStarts;
    std::vector<int> RowIndices;
    std::vector<double> Data;
};

void GatherBlockRows(const Eigen::SparseMatrix<double> &matrix, int block_column, std::vector<int> &rows) {
    rows.clear();
    for (int column = 0; column < 3; ++column)
        for (Eigen::SparseMatrix<double>::InnerIterator entry{matrix, 3 * block_column + column}; entry; ++entry)
            rows.push_back(int(entry.row()) / 3);
    std::ranges::sort(rows);
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
}

BlockSparseLower ToBlock3(const Eigen::SparseMatrix<double> &matrix) {
    if (matrix.rows() != matrix.cols() || matrix.rows() % 3 != 0) throw std::invalid_argument("Block3 sparse matrix dimensions must be square and divisible by three.");
    const int block_count = int(matrix.rows() / 3);
    BlockSparseLower result;
    result.ColumnStarts.resize(block_count + 1);
    result.RowIndices.reserve(matrix.nonZeros() / 3);
    result.Data.reserve(size_t(matrix.nonZeros() / 3) * 9);
    std::vector<int> rows, row_to_local(block_count, -1);
    std::vector<std::array<double, 9>> blocks;
    for (int block_column = 0; block_column < block_count; ++block_column) {
        GatherBlockRows(matrix, block_column, rows);
        blocks.assign(rows.size(), {});
        for (size_t index = 0; index < rows.size(); ++index) row_to_local[rows[index]] = int(index);
        for (int column = 0; column < 3; ++column) {
            for (Eigen::SparseMatrix<double>::InnerIterator entry{matrix, 3 * block_column + column}; entry; ++entry) {
                const int block_row = int(entry.row()) / 3, row = int(entry.row()) % 3;
                auto &block = blocks[row_to_local[block_row]];
                block[row + 3 * column] = entry.value();
                if (block_row == block_column) block[column + 3 * row] = entry.value();
            }
        }
        result.RowIndices.insert(result.RowIndices.end(), rows.begin(), rows.end());
        for (const auto &block : blocks) result.Data.insert(result.Data.end(), block.begin(), block.end());
        result.ColumnStarts[block_column + 1] = long(result.RowIndices.size());
    }
    return result;
}

struct SparsePattern {
    int Rows{}, Columns{};
    uint8_t BlockSize{};
    std::vector<long> ColumnStarts;
    std::vector<int> RowIndices;
};

bool Matches(const Eigen::SparseMatrix<double> &matrix, const SparsePattern &pattern) {
    if (!matrix.isCompressed() || matrix.rows() != matrix.cols() ||
        matrix.rows() != Eigen::Index(pattern.Rows) * pattern.BlockSize)
        return false;
    if (pattern.BlockSize == 1) {
        if (matrix.nonZeros() != Eigen::Index(pattern.RowIndices.size())) return false;
        for (Eigen::Index column = 0; column <= matrix.cols(); ++column)
            if (matrix.outerIndexPtr()[column] != pattern.ColumnStarts[column]) return false;
        return std::equal(
            pattern.RowIndices.begin(), pattern.RowIndices.end(), matrix.innerIndexPtr()
        );
    }
    if (pattern.BlockSize != 3) return false;

    std::vector<int> rows;
    size_t offset{};
    for (int block_column = 0; block_column < pattern.Columns; ++block_column) {
        GatherBlockRows(matrix, block_column, rows);
        const size_t end = offset + rows.size();
        if (end > pattern.RowIndices.size() || pattern.ColumnStarts[block_column] != long(offset) ||
            !std::equal(rows.begin(), rows.end(), pattern.RowIndices.begin() + offset))
            return false;
        offset = end;
    }
    return offset == pattern.RowIndices.size() && pattern.ColumnStarts.back() == long(offset);
}

struct SparseMatrixView {
    std::vector<long> ScalarColumnStarts;
    BlockSparseLower Block;
    SparseMatrix_Double Matrix{};

    SparseMatrixView(const Eigen::SparseMatrix<double> &matrix, SparseStorage storage) {
        if (matrix.rows() != matrix.cols() || !matrix.isCompressed()) throw std::invalid_argument("Sparse Cholesky requires a square compressed matrix.");
        if (storage == SparseStorage::Scalar) {
            ScalarColumnStarts.resize(matrix.cols() + 1);
            std::copy_n(matrix.outerIndexPtr(), ScalarColumnStarts.size(), ScalarColumnStarts.begin());
        } else {
            Block = ToBlock3(matrix);
        }
        const bool blocked = storage == SparseStorage::Block3;
        Matrix = {
            .structure = {
                .rowCount = blocked ? int(matrix.rows() / 3) : int(matrix.rows()),
                .columnCount = blocked ? int(matrix.cols() / 3) : int(matrix.cols()),
                .columnStarts = blocked ? Block.ColumnStarts.data() : ScalarColumnStarts.data(),
                .rowIndices = blocked ? Block.RowIndices.data() : const_cast<int *>(matrix.innerIndexPtr()),
                .attributes = {.transpose = false, .triangle = SparseLowerTriangle, .kind = SparseSymmetric, ._reserved = 0, ._allocatedBySparse = false},
                .blockSize = uint8_t(blocked ? 3 : 1),
            },
            .data = blocked ? Block.Data.data() : const_cast<double *>(matrix.valuePtr()),
        };
    }

    SparsePattern Pattern() const {
        const auto &structure = Matrix.structure;
        const long entries = structure.columnStarts[structure.columnCount];
        return {
            .Rows = structure.rowCount,
            .Columns = structure.columnCount,
            .BlockSize = structure.blockSize,
            .ColumnStarts = {structure.columnStarts, structure.columnStarts + structure.columnCount + 1},
            .RowIndices = {structure.rowIndices, structure.rowIndices + entries},
        };
    }
};

SparseSymbolicFactorOptions SymbolicOptions(SparseOrdering ordering) {
    return {
        .control = SparseDefaultControl,
        .orderMethod = ordering == SparseOrdering::Metis ? SparseOrderMetis : SparseOrderDefault,
        .malloc = std::malloc,
        .free = std::free,
    };
}

SparseNumericFactorOptions NumericOptions() {
    return {
        .control = SparseDefaultControl,
        .scalingMethod = SparseScalingDefault,
        .pivotTolerance = 0.01,
        .zeroTolerance = 1e-4 * DBL_EPSILON,
    };
}
} // namespace

struct SparseCholeskySymbolic::Factorization {
    SparseOpaqueSymbolicFactorization Opaque;
    SparsePattern Pattern;
    ~Factorization() { SparseCleanup(Opaque); }
};

struct SparseCholesky::Factorization {
    SparseOpaqueFactorization_Double Opaque;
    ~Factorization() { SparseCleanup(Opaque); }
};

struct SparseCholeskyCache::State {
    std::mutex Mutex;
    std::shared_ptr<SparseCholeskySymbolic> Symbolic;
    SparseOrdering Ordering{};
    SparseStorage Storage{};
};

SparseCholeskySymbolic::SparseCholeskySymbolic(const Eigen::SparseMatrix<double> &matrix, SparseOrdering ordering, SparseStorage storage) : Size(int(matrix.rows())) {
    ConfigureSparseExecution();
    const SparseMatrixView view{matrix, storage};
    auto opaque = SparseFactor(SparseFactorizationCholesky, view.Matrix.structure, SymbolicOptions(ordering));
    if (opaque.status != SparseStatusOK) {
        SparseCleanup(opaque);
        throw std::runtime_error("Sparse Cholesky symbolic factorization failed.");
    }
    Factor = std::make_unique<Factorization>(std::move(opaque), view.Pattern());
}

SparseCholeskySymbolic::~SparseCholeskySymbolic() = default;

bool SparseCholeskySymbolic::Matches(const Eigen::SparseMatrix<double> &matrix) const {
    return matrix.rows() == Size && ::Matches(matrix, Factor->Pattern);
}

SparseCholeskyCache::SparseCholeskyCache() : Cache(std::make_unique<State>()) {}

SparseCholeskyCache::~SparseCholeskyCache() = default;

SparseCholeskyCache::Acquisition SparseCholeskyCache::Acquire(
    const Eigen::SparseMatrix<double> &matrix, SparseOrdering ordering, SparseStorage storage
) {
    const std::lock_guard lock{Cache->Mutex};
    if (Cache->Symbolic && Cache->Ordering == ordering && Cache->Storage == storage && Cache->Symbolic->Matches(matrix))
        return {Cache->Symbolic, true};
    Cache->Symbolic = std::make_shared<SparseCholeskySymbolic>(matrix, ordering, storage);
    Cache->Ordering = ordering;
    Cache->Storage = storage;
    return {Cache->Symbolic, false};
}

SparseCholesky::SparseCholesky(const Eigen::SparseMatrix<double> &matrix, SparseOrdering ordering, SparseStorage storage) : Size(int(matrix.rows())) {
    ConfigureSparseExecution();
    const SparseMatrixView view{matrix, storage};
    SparseOpaqueFactorization_Double opaque;
    if (ordering == SparseOrdering::Default) {
        opaque = SparseFactor(SparseFactorizationCholesky, view.Matrix);
    } else {
        opaque = SparseFactor(SparseFactorizationCholesky, view.Matrix, SymbolicOptions(ordering), NumericOptions());
    }
    if (opaque.status != SparseStatusOK) {
        SparseCleanup(opaque);
        throw std::runtime_error("Sparse Cholesky factorization failed.");
    }
    Factor = std::make_unique<Factorization>(std::move(opaque));
}

SparseCholesky::SparseCholesky(const Eigen::SparseMatrix<double> &matrix, const SparseCholeskySymbolic &symbolic) : Size(int(matrix.rows())) {
    ConfigureSparseExecution();
    if (Size != symbolic.Size) throw std::invalid_argument("Sparse Cholesky symbolic factorization size mismatch.");
    const SparseMatrixView view{matrix, symbolic.Factor->Pattern.BlockSize == 3 ? SparseStorage::Block3 : SparseStorage::Scalar};
    if (!::Matches(matrix, symbolic.Factor->Pattern))
        throw std::invalid_argument("Sparse Cholesky symbolic factorization sparsity mismatch.");
    auto opaque = SparseFactor(symbolic.Factor->Opaque, view.Matrix, NumericOptions());
    if (opaque.status != SparseStatusOK) {
        SparseCleanup(opaque);
        throw std::runtime_error("Sparse Cholesky numeric factorization failed.");
    }
    Factor = std::make_unique<Factorization>(std::move(opaque));
}

SparseCholesky::~SparseCholesky() = default;

void SparseCholesky::Solve(const double *input, double *output, int width) const {
    const DenseMatrix_Double b{Size, width, Size, SparseAttributes_t{}, const_cast<double *>(input)};
    const DenseMatrix_Double x{Size, width, Size, SparseAttributes_t{}, output};
    SparseSolve(Factor->Opaque, b, x);
}
