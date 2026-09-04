#include "AccelerateSparseCholesky.h"

#include "SparseExecution.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <stdexcept>
#include <vector>

namespace {
constexpr int BlockCrossover{8192};

struct BlockSparseLower {
    std::vector<long> ColumnStarts;
    std::vector<int> RowIndices;
    std::vector<double> Data;
};

void GatherBlockRows(const numeric::SparseMatrix &matrix, int block_column, std::vector<int> &rows) {
    rows.clear();
    for (int column = 0; column < 3; ++column)
        for (long entry = matrix.ColumnStarts[3 * block_column + column]; entry < matrix.ColumnStarts[3 * block_column + column + 1]; ++entry)
            rows.push_back(matrix.RowIndices[entry] / 3);
    std::ranges::sort(rows);
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
}

BlockSparseLower ToBlock3(const numeric::SparseMatrix &matrix) {
    if (matrix.rows() != matrix.cols() || matrix.rows() % 3 != 0) throw std::invalid_argument("Block3 sparse matrix dimensions must be square and divisible by three.");
    const int block_count = int(matrix.rows() / 3);
    BlockSparseLower result;
    result.ColumnStarts.resize(block_count + 1);
    result.RowIndices.reserve(matrix.NonZeros() / 3);
    result.Data.reserve(matrix.NonZeros() / 3 * 9);
    std::vector<int> rows, row_to_local(block_count, -1);
    std::vector<std::array<double, 9>> blocks;
    for (int block_column = 0; block_column < block_count; ++block_column) {
        GatherBlockRows(matrix, block_column, rows);
        blocks.assign(rows.size(), {});
        for (size_t index = 0; index < rows.size(); ++index) row_to_local[rows[index]] = int(index);
        for (int column = 0; column < 3; ++column) {
            const int scalar_column = 3 * block_column + column;
            for (long entry = matrix.ColumnStarts[scalar_column]; entry < matrix.ColumnStarts[scalar_column + 1]; ++entry) {
                const int block_row = matrix.RowIndices[entry] / 3, row = matrix.RowIndices[entry] % 3;
                auto &block = blocks[row_to_local[block_row]];
                block[row + 3 * column] = matrix.Values[entry];
                if (block_row == block_column) block[column + 3 * row] = matrix.Values[entry];
            }
        }
        result.RowIndices.insert(result.RowIndices.end(), rows.begin(), rows.end());
        for (const auto &block : blocks) result.Data.insert(result.Data.end(), block.begin(), block.end());
        result.ColumnStarts[block_column + 1] = long(result.RowIndices.size());
    }
    return result;
}

struct SparseMatrixView {
    BlockSparseLower Block;
    SparseMatrix_Double Matrix{};

    SparseMatrixView(const numeric::SparseMatrix &matrix, bool blocked) {
        if (matrix.rows() != matrix.cols()) throw std::invalid_argument("Sparse Cholesky requires a square compressed matrix.");
        if (blocked) Block = ToBlock3(matrix);
        Matrix = {
            .structure = {
                .rowCount = blocked ? int(matrix.rows() / 3) : int(matrix.rows()),
                .columnCount = blocked ? int(matrix.cols() / 3) : int(matrix.cols()),
                .columnStarts = blocked ? Block.ColumnStarts.data() : const_cast<long *>(matrix.ColumnStarts.data()),
                .rowIndices = blocked ? Block.RowIndices.data() : const_cast<int *>(matrix.RowIndices.data()),
                .attributes = {.transpose = false, .triangle = SparseLowerTriangle, .kind = SparseSymmetric, ._reserved = 0, ._allocatedBySparse = false},
                .blockSize = uint8_t(blocked ? 3 : 1),
            },
            .data = blocked ? Block.Data.data() : const_cast<double *>(matrix.Values.data()),
        };
    }
};

SparseSymbolicFactorOptions SymbolicOptions(bool use_metis) {
    return {
        .control = SparseDefaultControl,
        .orderMethod = use_metis ? SparseOrderMetis : SparseOrderDefault,
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

bool UseBlock3(const numeric::SparseMatrix &matrix) {
    return matrix.rows() >= BlockCrossover && matrix.rows() % 3 == 0;
}
} // namespace

struct modal::AccelerateSparseCholesky::Factorization {
    SparseOpaqueFactorization_Double Opaque;
    int Size{};
    ~Factorization() { SparseCleanup(Opaque); }
};

modal::AccelerateSparseCholesky::AccelerateSparseCholesky(const numeric::SparseMatrix &matrix) {
    ConfigureAccelerateSparseExecution();
    const bool large = matrix.rows() >= BlockCrossover;
    const bool blocked = UseBlock3(matrix);
    const SparseMatrixView view{matrix, blocked};
    SparseOpaqueFactorization_Double opaque;
    if (!large) {
        opaque = SparseFactor(SparseFactorizationCholesky, view.Matrix);
    } else {
        opaque = SparseFactor(SparseFactorizationCholesky, view.Matrix, SymbolicOptions(true), NumericOptions());
    }
    if (opaque.status != SparseStatusOK) {
        SparseCleanup(opaque);
        throw std::runtime_error("Sparse Cholesky factorization failed.");
    }
    Factor = std::make_unique<Factorization>(std::move(opaque), int(matrix.rows()));
}

modal::AccelerateSparseCholesky::~AccelerateSparseCholesky() = default;

void modal::AccelerateSparseCholesky::Solve(const double *input, double *output, int width) const {
    if (width < 1) throw std::invalid_argument("Sparse Cholesky solve width must be positive.");
    const DenseMatrix_Double b{Factor->Size, width, Factor->Size, SparseAttributes_t{}, const_cast<double *>(input)};
    const DenseMatrix_Double x{Factor->Size, width, Factor->Size, SparseAttributes_t{}, output};
    SparseSolve(Factor->Opaque, b, x);
}
