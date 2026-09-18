#pragma once

#include "numeric/Vector.h"

#include "numeric/Accelerate.h"
#include "numeric/Matrix.h"
#include "numeric/SparseMatrix.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace modal::eigensolver {
using numeric::GeneralizedSelfAdjointEigenSolve, numeric::Matrix, numeric::MatrixView, numeric::SelfAdjointEigenSolve, numeric::SparseMatrix, numeric::Vector, numeric::VectorView;

struct GeneralizedEigenOptions {
    uint32_t Count{};
    uint32_t CertifiedCount{};
    uint32_t SubspaceSize{};
    double Shift{};
    double IterationTolerance{1e-4};
    double ResidualTolerance{1e-8};
    uint32_t MaxIterations{100};
    uint32_t MaxRefinementIterations{10};
    uint64_t RandomSeed{20260710};
    uint32_t KrylovBlockWidth{1};
    uint32_t KrylovSize{};
    uint32_t ExtendedKrylovSize{};
    double ExtensionResidual{1e-3};
};

struct GeneralizedEigenControl {
    const std::atomic<bool> *Cancel{};
    std::atomic<float> *Progress{};
    float ProgressBegin{};
    float ProgressEnd{1};
};

struct GeneralizedEigenResult {
    Vector<double> Eigenvalues;
    Matrix<double> Eigenvectors;
    Vector<double> RelativeResiduals;
    double MassOrthogonalityError{};
    uint32_t Iterations{}, OpApplications{};
    bool Converged{};
};

inline bool SelfAdjointEigenvectors(Matrix<double> &matrix, Vector<double> &values) {
    if (matrix.rows() != matrix.cols()) return false;
    values.Resize(matrix.rows());
    return SelfAdjointEigenSolve(matrix.data(), values.data(), uint32_t(matrix.rows()));
}

inline bool GeneralizedEigenvectors(
    Matrix<double> &stiffness, Matrix<double> &mass, Vector<double> &values
) {
    if (stiffness.rows() != stiffness.cols() || mass.rows() != stiffness.rows() || mass.cols() != stiffness.cols()) return false;
    values.Resize(stiffness.rows());
    return GeneralizedSelfAdjointEigenSolve(stiffness.data(), mass.data(), values.data(), uint32_t(stiffness.rows()));
}

inline void ScaleRows(MatrixView<double> matrix, VectorView<const double> scales) {
    for (size_t column = 0; column < matrix.Columns; ++column)
        for (size_t row = 0; row < matrix.Rows; ++row) matrix(row, column) *= scales[row];
}

inline Vector<double> InverseSqrtDiagonal(MatrixView<const double> matrix) {
    Vector<double> result(matrix.Rows);
    for (size_t index = 0; index < matrix.Rows; ++index) result[index] = 1 / std::sqrt(matrix(index, index));
    return result;
}

inline void Subtract(MatrixView<double> destination, MatrixView<const double> source) {
    AddScaled(-1, source, destination);
}

inline void CertifyGeneralizedEigenResult(
    GeneralizedEigenResult &result, MatrixView<const double> mass_vectors,
    MatrixView<const double> stiffness_vectors, uint32_t certified_count,
    double rigid_threshold, double residual_tolerance
) {
    const uint32_t count = uint32_t(result.Eigenvalues.size());
    const Matrix<double> residual = ColumnScaledDifference(stiffness_vectors, mass_vectors, result.Eigenvalues.View());
    result.RelativeResiduals.Resize(count);
    double maximum_physical_residual{};
    for (uint32_t mode = 0; mode < count; ++mode) {
        const double scale = Norm(stiffness_vectors.Column(mode)) +
            std::abs(result.Eigenvalues[mode]) * Norm(mass_vectors.Column(mode));
        const double norm = Norm(residual.Column(mode));
        result.RelativeResiduals[mode] = scale == 0 ? norm : norm / scale;
        if (mode < certified_count && std::abs(result.Eigenvalues[mode]) > rigid_threshold)
            maximum_physical_residual = std::max(maximum_physical_residual, result.RelativeResiduals[mode]);
    }
    Matrix<double> gram = TransposeMultiply(result.Eigenvectors.View(), mass_vectors);
    for (uint32_t mode = 0; mode < count; ++mode) gram(mode, mode) -= 1;
    result.MassOrthogonalityError = Norm(gram.View());
    result.Converged = maximum_physical_residual <= residual_tolerance &&
        result.MassOrthogonalityError <= std::max(1e-9, 10 * residual_tolerance);
}

template<class ShiftInvert>
uint32_t RefineGeneralizedEigenpairs(
    const ShiftInvert &operation, const SparseMatrix &mass,
    const SparseMatrix &stiffness, double target_residual,
    double rigid_threshold, uint32_t certified_count, uint32_t max_iterations,
    Vector<double> &eigenvalues, Matrix<double> &eigenvectors,
    Matrix<double> &mass_vectors, Matrix<double> &stiffness_vectors,
    bool actions_initialized = false
) {
    constexpr double ClusterRelativeGap{1e-3};
    const uint32_t width = uint32_t(eigenvalues.size());
    uint32_t applications{};
    if (!actions_initialized) {
        mass_vectors = SymmetricMultiply(mass, eigenvectors.View());
        stiffness_vectors = SymmetricMultiply(stiffness, eigenvectors.View());
    }
    for (uint32_t iteration = 0; iteration < max_iterations; ++iteration) {
        const Matrix<double> residual = ColumnScaledDifference(stiffness_vectors.View(), mass_vectors.View(), eigenvalues.View());
        std::vector<bool> active(width);
        bool unconverged{};
        for (uint32_t mode = 0; mode < std::min(certified_count, width); ++mode) {
            if (std::abs(eigenvalues[mode]) <= rigid_threshold) continue;
            const double scale = Norm(stiffness_vectors.Column(mode)) +
                std::abs(eigenvalues[mode]) * Norm(mass_vectors.Column(mode));
            if (Norm(residual.Column(mode)) > target_residual * scale) {
                unconverged = true;
                active[mode] = true;
            }
        }
        if (!unconverged) break;
        bool expanded;
        do {
            expanded = false;
            for (uint32_t mode = 0; mode + 1 < width; ++mode) {
                if (active[mode] == active[mode + 1] || std::abs(eigenvalues[mode]) <= rigid_threshold ||
                    std::abs(eigenvalues[mode + 1]) <= rigid_threshold)
                    continue;
                const double scale = std::max({std::abs(eigenvalues[mode]), std::abs(eigenvalues[mode + 1]), rigid_threshold});
                if (std::abs(eigenvalues[mode + 1] - eigenvalues[mode]) <= ClusterRelativeGap * scale) {
                    active[mode] = active[mode + 1] = true;
                    expanded = true;
                }
            }
        } while (expanded);
        std::vector<uint32_t> active_modes;
        for (uint32_t mode = 0; mode < width; ++mode)
            if (active[mode]) active_modes.push_back(mode);
        if (active_modes.empty()) break;

        Matrix<double> active_residual(eigenvectors.rows(), active_modes.size());
        for (size_t column = 0; column < active_modes.size(); ++column)
            Copy(residual.Column(active_modes[column]), active_residual.Column(column));
        Matrix<double> correction(eigenvectors.rows(), active_modes.size());
        operation.solve_panel(active_residual.data(), correction.data(), int(active_modes.size()));
        applications += uint32_t(active_modes.size());

        Matrix<double> mass_correction = SymmetricMultiply(mass, correction.View());
        for (uint32_t pass = 0; pass < 2; ++pass) {
            const Matrix<double> coefficients = TransposeMultiply(eigenvectors.View(), mass_correction.View());
            SubtractProduct(correction.View(), eigenvectors.View(), coefficients.View());
            SubtractProduct(mass_correction.View(), mass_vectors.View(), coefficients.View());
        }
        Matrix<double> gram = TransposeMultiply(correction.View(), mass_correction.View());
        Symmetrize(gram.View());
        Vector<double> correction_values;
        if (!SelfAdjointEigenvectors(gram, correction_values)) break;
        const double threshold = correction_values[correction_values.size() - 1] * 1e-12;
        size_t first{};
        while (first < correction.cols() && correction_values[first] <= threshold) ++first;
        if (first == correction.cols()) break;
        const size_t correction_width = correction.cols() - first;
        Matrix<double> transform = Copy(gram.LastColumns(correction_width));
        Vector<double> inverse_sqrt(correction_width);
        for (size_t column = 0; column < correction_width; ++column)
            inverse_sqrt[column] = 1 / std::sqrt(correction_values[first + column]);
        ScaleColumns(transform.View(), inverse_sqrt.View());
        correction = Multiply(correction.View(), transform.View());
        mass_correction = Multiply(mass_correction.View(), transform.View());
        const Matrix<double> stiffness_correction = SymmetricMultiply(stiffness, correction.View());

        Matrix<double> space(eigenvectors.rows(), width + correction_width);
        Copy(eigenvectors.View(), space.FirstColumns(width));
        Copy(correction.View(), space.LastColumns(correction_width));
        Matrix<double> mass_space(eigenvectors.rows(), width + correction_width);
        Copy(mass_vectors.View(), mass_space.FirstColumns(width));
        Copy(mass_correction.View(), mass_space.LastColumns(correction_width));
        Matrix<double> stiffness_space(eigenvectors.rows(), width + correction_width);
        Copy(stiffness_vectors.View(), stiffness_space.FirstColumns(width));
        Copy(stiffness_correction.View(), stiffness_space.LastColumns(correction_width));
        Matrix<double> projected = TransposeMultiply(space.View(), stiffness_space.View());
        Symmetrize(projected.View());
        Vector<double> projected_values;
        if (!SelfAdjointEigenvectors(projected, projected_values)) break;
        const auto rotation = projected.FirstColumns(width);
        eigenvalues = Copy(projected_values.First(width));
        eigenvectors = Multiply(space.View(), rotation);
        mass_vectors = Multiply(mass_space.View(), rotation);
        stiffness_vectors = Multiply(stiffness_space.View(), rotation);
    }
    return applications;
}

template<class ShiftInvert>
GeneralizedEigenResult SolveGeneralizedInverseIteration(
    ShiftInvert &operation, const SparseMatrix &mass,
    const SparseMatrix &stiffness, const GeneralizedEigenOptions &options,
    const Matrix<float> *seed = nullptr, GeneralizedEigenControl control = {}
) {
    const uint32_t n = uint32_t(mass.rows());
    const uint32_t count = std::min(options.Count, n > 0 ? n - 1 : 0);
    const uint32_t subspace_size = std::min(std::max(options.SubspaceSize, count + 1), n);
    GeneralizedEigenResult result;
    if (!count || subspace_size <= count || stiffness.rows() != mass.rows() || stiffness.cols() != mass.cols()) return result;

    bool operation_ready{};
    if (seed && seed->rows() == n && seed->cols() >= count) {
        Matrix<double> seed_vectors = Cast<double>(seed->FirstColumns(count));
        Matrix<double> seed_mass = SymmetricMultiply(mass, seed_vectors.View());
        Matrix<double> seed_stiffness = SymmetricMultiply(stiffness, seed_vectors.View());
        Matrix<double> projected_mass = TransposeMultiply(seed_vectors.View(), seed_mass.View());
        Matrix<double> projected_stiffness = TransposeMultiply(seed_vectors.View(), seed_stiffness.View());
        Symmetrize(projected_mass.View());
        Symmetrize(projected_stiffness.View());
        Vector<double> seed_values;
        if (GeneralizedEigenvectors(projected_stiffness, projected_mass, seed_values)) {
            result.Eigenvalues = std::move(seed_values);
            result.Eigenvectors = Multiply(seed_vectors.View(), projected_stiffness.View());
            seed_mass = Multiply(seed_mass.View(), projected_stiffness.View());
            seed_stiffness = Multiply(seed_stiffness.View(), projected_stiffness.View());
            const uint32_t certified_count = options.CertifiedCount ? std::min(options.CertifiedCount, count) : count;
            const double rigid_threshold = std::max(std::abs(options.Shift) * 1e-4, 1e-12);
            CertifyGeneralizedEigenResult(result, seed_mass.View(), seed_stiffness.View(), certified_count, rigid_threshold, options.ResidualTolerance);
            if (result.Converged) return result;
            operation.set_shift(options.Shift);
            operation_ready = true;
            result.OpApplications += RefineGeneralizedEigenpairs(
                operation, mass, stiffness, options.ResidualTolerance, rigid_threshold,
                certified_count, std::min(2u, options.MaxRefinementIterations),
                result.Eigenvalues, result.Eigenvectors, seed_mass, seed_stiffness, true
            );
            CertifyGeneralizedEigenResult(result, seed_mass.View(), seed_stiffness.View(), certified_count, rigid_threshold, options.ResidualTolerance);
            if (result.Converged) return result;
            result = {};
        }
    }

    if (!operation_ready) operation.set_shift(options.Shift);
    const uint32_t seeded = seed && seed->rows() == n ? std::min(uint32_t(seed->cols()), subspace_size) : 0;
    Matrix<double> mass_space(n, subspace_size);
    {
        Matrix<double> space(n, subspace_size);
        std::mt19937_64 random{options.RandomSeed};
        if (seeded) {
            const auto converted = Cast<double>(seed->FirstColumns(seeded));
            Copy(converted.View(), space.FirstColumns(seeded));
        }
        for (uint32_t column = seeded; column < subspace_size; ++column)
            for (uint32_t row = 0; row < n; ++row) space(row, column) = (random() & 1) ? 1.0 : -1.0;
        SymmetricMultiply(mass, space.View(), mass_space.View());
    }

    Matrix<double> locked_vectors(n, count), locked_mass_vectors(n, count);
    Vector<double> locked_shifted_values(count);
    Vector<double> previous_values(count, std::numeric_limits<double>::max());
    uint32_t locked{};
    for (uint32_t iteration = 0; iteration < options.MaxIterations; ++iteration) {
        if (control.Cancel && control.Cancel->load(std::memory_order_relaxed)) return {};
        const uint32_t width = subspace_size - locked;
        Matrix<double> space(n, width);
        operation.solve_panel(mass_space.data(), space.data(), int(width));
        result.OpApplications += width;

        Matrix<double> mass_vectors = SymmetricMultiply(mass, space.View());
        Matrix<double> projected_shifted = TransposeMultiply(space.View(), mass_space.View());
        Matrix<double> projected_mass = TransposeMultiply(space.View(), mass_vectors.View());
        if (locked) {
            const auto locked_vectors_view = locked_vectors.FirstColumns(locked);
            const auto locked_mass_view = locked_mass_vectors.FirstColumns(locked);
            const Matrix<double> coefficients = TransposeMultiply(locked_vectors_view, mass_vectors.View());
            SubtractProduct(space.View(), locked_vectors_view, coefficients.View());
            SubtractProduct(mass_vectors.View(), locked_mass_view, coefficients.View());
            Matrix<double> weighted = Copy(coefficients.View());
            ScaleRows(weighted.View(), locked_shifted_values.First(locked));
            const Matrix<double> correction = TransposeMultiply(coefficients.View(), weighted.View());
            Subtract(projected_shifted.View(), correction.View());
            projected_mass = TransposeMultiply(space.View(), mass_vectors.View());
        }
        Symmetrize(projected_shifted.View());
        Symmetrize(projected_mass.View());
        const Vector<double> inverse_norm = InverseSqrtDiagonal(projected_mass.View());
        if (!AllFinite(inverse_norm.View())) return {};
        ScaleRowsAndColumns(projected_shifted.View(), inverse_norm.View());
        ScaleRowsAndColumns(projected_mass.View(), inverse_norm.View());
        Vector<double> decomposition_values;
        if (!GeneralizedEigenvectors(projected_shifted, projected_mass, decomposition_values)) return {};
        Matrix<double> rotation = std::move(projected_shifted);
        ScaleRows(rotation.View(), inverse_norm.View());

        uint32_t newly_locked{};
        for (uint32_t mode = 0; mode < width && locked + mode < count; ++mode) {
            const double value = decomposition_values[mode] + options.Shift;
            const double relative_change = std::abs(value - previous_values[locked + mode]) /
                std::max(std::abs(value), std::abs(options.Shift));
            previous_values[locked + mode] = value;
            if (newly_locked == mode && relative_change < options.IterationTolerance) ++newly_locked;
        }
        while (newly_locked && locked + newly_locked < count && newly_locked < width) {
            const double left = decomposition_values[newly_locked - 1] + options.Shift;
            const double right = decomposition_values[newly_locked] + options.Shift;
            const double scale = std::max({std::abs(left), std::abs(right), std::abs(options.Shift) * 1e-4});
            if (std::abs(right - left) > 1e-3 * scale) break;
            --newly_locked;
        }
        if (newly_locked) {
            Multiply(space.View(), rotation.FirstColumns(newly_locked), locked_vectors.ColumnsAt(locked, newly_locked));
            Multiply(mass_vectors.View(), rotation.FirstColumns(newly_locked), locked_mass_vectors.ColumnsAt(locked, newly_locked));
            Copy(decomposition_values.First(newly_locked), locked_shifted_values.Subvector(locked, newly_locked));
            locked += newly_locked;
        }
        result.Iterations = iteration + 1;
        if (control.Progress) {
            const float fraction = float(locked) / float(count);
            control.Progress->store(control.ProgressBegin + fraction * (control.ProgressEnd - control.ProgressBegin), std::memory_order_relaxed);
        }
        if (locked == count) {
            const double rigid_threshold = std::max(std::abs(options.Shift) * 1e-4, 1e-12);
            const uint32_t certified_count = options.CertifiedCount ? std::min(options.CertifiedCount, count) : count;
            if (seeded) {
                result.Eigenvalues = previous_values;
                result.Eigenvectors = locked_vectors;
                const Matrix<double> locked_stiffness = SymmetricMultiply(stiffness, locked_vectors.View());
                CertifyGeneralizedEigenResult(result, locked_mass_vectors.View(), locked_stiffness.View(), certified_count, rigid_threshold, options.ResidualTolerance);
                if (result.Converged) return result;
            }
            const uint32_t guard_count = width - newly_locked;
            Vector<double> refined_values(count + guard_count);
            Copy(previous_values.View(), refined_values.First(count));
            Matrix<double> refined_vectors(n, count + guard_count);
            Copy(locked_vectors.View(), refined_vectors.FirstColumns(count));
            if (guard_count) {
                for (uint32_t guard = 0; guard < guard_count; ++guard)
                    refined_values[count + guard] = decomposition_values[decomposition_values.size() - guard_count + guard] + options.Shift;
                Multiply(space.View(), rotation.LastColumns(guard_count), refined_vectors.LastColumns(guard_count));
            }
            Matrix<double> refined_mass, refined_stiffness;
            result.OpApplications += RefineGeneralizedEigenpairs(
                operation, mass, stiffness, options.ResidualTolerance, rigid_threshold, count,
                options.MaxRefinementIterations, refined_values, refined_vectors, refined_mass, refined_stiffness
            );
            result.Eigenvalues = Copy(refined_values.First(count));
            result.Eigenvectors = Copy(refined_vectors.FirstColumns(count));
            CertifyGeneralizedEigenResult(
                result, refined_mass.FirstColumns(count), refined_stiffness.FirstColumns(count),
                certified_count, rigid_threshold, options.ResidualTolerance
            );
            return result;
        }
        const uint32_t retained = width - newly_locked;
        mass_space.Resize(n, retained);
        Multiply(mass_vectors.View(), rotation.LastColumns(retained), mass_space.View());
    }
    return {};
}

template<class ShiftInvert>
GeneralizedEigenResult SolveGeneralizedBlockKrylov(
    ShiftInvert &operation, const SparseMatrix &mass,
    const SparseMatrix &stiffness, const GeneralizedEigenOptions &options,
    GeneralizedEigenControl control
) {
    const uint32_t block_width = std::max(options.KrylovBlockWidth, 1u);
    const uint32_t n = uint32_t(mass.rows());
    const uint32_t count = std::min(options.Count, n > 0 ? n - 1 : 0);
    const uint32_t retained_size = std::min(std::max(options.SubspaceSize, count + 1), n);
    const uint32_t default_size = std::max({retained_size, 2 * count + 6, count + 40});
    const uint32_t subspace_size = std::min(std::max(options.KrylovSize, default_size), n);
    const uint32_t capacity = std::min(std::max(options.ExtendedKrylovSize, subspace_size), n);
    GeneralizedEigenResult result;
    if (!count || subspace_size <= count || stiffness.rows() != mass.rows() || stiffness.cols() != mass.cols()) return result;

    operation.set_shift(options.Shift);
    Matrix<double> basis(n, capacity), mass_basis(n, capacity), action_basis(n, capacity);
    const auto append = [&](Matrix<double> vectors, Matrix<double> mass_vectors, uint32_t used) {
        if (block_width == 1) {
            for (uint32_t pass = 0; pass < 2 && used; ++pass) {
                const Matrix<double> coefficients = TransposeMultiply(basis.FirstColumns(used), mass_vectors.View());
                SubtractProduct(vectors.View(), basis.FirstColumns(used), coefficients.View());
                SubtractProduct(mass_vectors.View(), mass_basis.FirstColumns(used), coefficients.View());
            }
        } else {
            constexpr uint32_t OrthogonalizationBlock{16};
            for (uint32_t first = 0; first < used; first += OrthogonalizationBlock) {
                const uint32_t width = std::min(OrthogonalizationBlock, used - first);
                const Matrix<double> coefficients = TransposeMultiply(basis.ColumnsAt(first, width), mass_vectors.View());
                SubtractProduct(vectors.View(), basis.ColumnsAt(first, width), coefficients.View());
                SubtractProduct(mass_vectors.View(), mass_basis.ColumnsAt(first, width), coefficients.View());
            }
        }
        Matrix<double> gram = TransposeMultiply(vectors.View(), mass_vectors.View());
        const Vector<double> inverse_norm = InverseSqrtDiagonal(gram.View());
        if (!AllFinite(inverse_norm.View())) return uint32_t{};
        Symmetrize(gram.View());
        ScaleRowsAndColumns(gram.View(), inverse_norm.View());
        Vector<double> values;
        if (!SelfAdjointEigenvectors(gram, values)) return uint32_t{};
        double maximum{};
        for (double value : values.Values) maximum = std::max(maximum, std::abs(value));
        const double threshold = maximum * 1e-12;
        size_t first{};
        while (first < values.size() && values[first] <= threshold) ++first;
        const uint32_t rank = uint32_t(values.size() - first);
        const uint32_t retained = std::min(rank, capacity - used);
        if (!retained) return uint32_t{};
        Matrix<double> transform = Copy(gram.ColumnsAt(first, retained));
        ScaleRows(transform.View(), inverse_norm.View());
        Vector<double> inverse_sqrt(retained);
        for (uint32_t column = 0; column < retained; ++column) inverse_sqrt[column] = 1 / std::sqrt(values[first + column]);
        ScaleColumns(transform.View(), inverse_sqrt.View());
        Multiply(vectors.View(), transform.View(), basis.ColumnsAt(used, retained));
        Multiply(mass_vectors.View(), transform.View(), mass_basis.ColumnsAt(used, retained));
        return retained;
    };

    const uint32_t initial_width = std::min(block_width, capacity);
    Matrix<double> initial(n, initial_width);
    std::mt19937_64 random{options.RandomSeed};
    for (uint32_t column = 0; column < initial_width; ++column)
        for (uint32_t row = 0; row < n; ++row) initial(row, column) = (random() & 1) ? 1.0 : -1.0;
    Matrix<double> initial_mass = SymmetricMultiply(mass, initial.View());
    uint32_t used = append(std::move(initial), std::move(initial_mass), 0);
    uint32_t applied{};
    const double rigid_threshold = std::max(std::abs(options.Shift) * 1e-4, 1e-12);
    const uint32_t certified_count = options.CertifiedCount ? std::min(options.CertifiedCount, count) : count;
    const auto grow = [&](uint32_t target) {
        while (applied < target) {
            if (control.Cancel && control.Cancel->load(std::memory_order_relaxed)) return false;
            if (applied >= used) return false;
            const uint32_t width = std::min(block_width, used - applied);
            operation.solve_panel(mass_basis.Column(applied).data(), action_basis.Column(applied).data(), int(width));
            result.OpApplications += width;
            if (used < capacity) {
                Matrix<double> next = Copy(action_basis.ColumnsAt(applied, width));
                Matrix<double> next_mass = SymmetricMultiply(mass, next.View());
                const uint32_t added = append(std::move(next), std::move(next_mass), used);
                if (!added) return false;
                used += added;
            }
            applied += width;
            ++result.Iterations;
            if (control.Progress) {
                const float fraction = float(applied) / float(capacity);
                control.Progress->store(control.ProgressBegin + fraction * (control.ProgressEnd - control.ProgressBegin), std::memory_order_relaxed);
            }
        }
        return true;
    };
    Vector<double> values;
    Matrix<double> vectors, mass_vectors, stiffness_vectors;
    const auto extract = [&](uint32_t width) {
        Matrix<double> projected = TransposeMultiply(mass_basis.FirstColumns(width), action_basis.FirstColumns(width));
        Symmetrize(projected.View());
        Matrix<double> projected_mass = TransposeMultiply(basis.FirstColumns(width), mass_basis.FirstColumns(width));
        Symmetrize(projected_mass.View());
        Vector<double> inverse_values;
        if (!GeneralizedEigenvectors(projected, projected_mass, inverse_values)) return false;
        Matrix<double> rotation(width, retained_size);
        values.Resize(retained_size);
        for (uint32_t mode = 0; mode < retained_size; ++mode) {
            const size_t index = inverse_values.size() - 1 - mode;
            const double inverse_value = inverse_values[index];
            if (!(inverse_value > 0) || !std::isfinite(inverse_value)) return false;
            values[mode] = options.Shift + 1 / inverse_value;
            Copy(projected.Column(index), rotation.Column(mode));
        }
        vectors.Resize(n, retained_size);
        mass_vectors.Resize(n, retained_size);
        Multiply(basis.FirstColumns(width), rotation.View(), vectors.View());
        Multiply(mass_basis.FirstColumns(width), rotation.View(), mass_vectors.View());
        stiffness_vectors = SymmetricMultiply(stiffness, vectors.View());
        result.Eigenvalues = Copy(values.First(count));
        result.Eigenvectors = Copy(vectors.FirstColumns(count));
        CertifyGeneralizedEigenResult(result, mass_vectors.FirstColumns(count), stiffness_vectors.FirstColumns(count), certified_count, rigid_threshold, options.ResidualTolerance);
        return true;
    };
    if (!grow(subspace_size) || !extract(subspace_size)) return {};
    double maximum_residual{};
    for (uint32_t mode = 0; mode < certified_count; ++mode)
        if (std::abs(values[mode]) > rigid_threshold) maximum_residual = std::max(maximum_residual, result.RelativeResiduals[mode]);
    if (capacity > subspace_size && maximum_residual > options.ExtensionResidual)
        if (!grow(capacity) || !extract(capacity)) return {};
    result.OpApplications += RefineGeneralizedEigenpairs(
        operation, mass, stiffness, options.ResidualTolerance, rigid_threshold, count,
        options.MaxRefinementIterations, values, vectors, mass_vectors, stiffness_vectors, block_width > 1
    );
    result.Eigenvalues = Copy(values.First(count));
    result.Eigenvectors = Copy(vectors.FirstColumns(count));
    CertifyGeneralizedEigenResult(result, mass_vectors.FirstColumns(count), stiffness_vectors.FirstColumns(count), certified_count, rigid_threshold, options.ResidualTolerance);
    if (control.Progress) control.Progress->store(control.ProgressEnd, std::memory_order_relaxed);
    return result;
}

template<class ShiftInvert>
GeneralizedEigenResult SolveGeneralizedEigenproblem(
    ShiftInvert &operation, const SparseMatrix &mass,
    const SparseMatrix &stiffness, const GeneralizedEigenOptions &options,
    const Matrix<float> *seed = nullptr, GeneralizedEigenControl control = {}
) {
    if (seed || options.Count <= 12 || options.Count >= 128 || mass.rows() >= 100000)
        return SolveGeneralizedInverseIteration(operation, mass, stiffness, options, seed, control);
    auto result = SolveGeneralizedBlockKrylov(operation, mass, stiffness, options, control);
    if (result.Converged) return result;
    return SolveGeneralizedInverseIteration(operation, mass, stiffness, options, seed, control);
}
} // namespace modal::eigensolver
