#pragma once

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
    numeric::Vector<double> Eigenvalues;
    numeric::Matrix<double> Eigenvectors;
    numeric::Vector<double> RelativeResiduals;
    double MassOrthogonalityError{};
    uint32_t Iterations{}, OpApplications{};
    bool Converged{};
};

inline bool SelfAdjointEigenvectors(numeric::Matrix<double> &matrix, numeric::Vector<double> &values) {
    if (matrix.rows() != matrix.cols()) return false;
    values.Resize(matrix.rows());
    return numeric::SelfAdjointEigenSolve(matrix.data(), values.data(), uint32_t(matrix.rows()));
}

inline bool GeneralizedEigenvectors(
    numeric::Matrix<double> &stiffness, numeric::Matrix<double> &mass, numeric::Vector<double> &values
) {
    if (stiffness.rows() != stiffness.cols() || mass.rows() != stiffness.rows() || mass.cols() != stiffness.cols()) return false;
    values.Resize(stiffness.rows());
    return numeric::GeneralizedSelfAdjointEigenSolve(stiffness.data(), mass.data(), values.data(), uint32_t(stiffness.rows()));
}

inline void ScaleRows(numeric::MatrixView<double> matrix, numeric::VectorView<const double> scales) {
    for (size_t column = 0; column < matrix.Columns; ++column)
        for (size_t row = 0; row < matrix.Rows; ++row) matrix(row, column) *= scales[row];
}

inline numeric::Vector<double> InverseSqrtDiagonal(numeric::MatrixView<const double> matrix) {
    numeric::Vector<double> result(matrix.Rows);
    for (size_t index = 0; index < matrix.Rows; ++index) result[index] = 1 / std::sqrt(matrix(index, index));
    return result;
}

inline void Subtract(numeric::MatrixView<double> destination, numeric::MatrixView<const double> source) {
    numeric::AddScaled(-1, source, destination);
}

inline void CertifyGeneralizedEigenResult(
    GeneralizedEigenResult &result, numeric::MatrixView<const double> mass_vectors,
    numeric::MatrixView<const double> stiffness_vectors, uint32_t certified_count,
    double rigid_threshold, double residual_tolerance
) {
    const uint32_t count = uint32_t(result.Eigenvalues.size());
    const numeric::Matrix<double> residual = numeric::ColumnScaledDifference(stiffness_vectors, mass_vectors, result.Eigenvalues.View());
    result.RelativeResiduals.Resize(count);
    double maximum_physical_residual{};
    for (uint32_t mode = 0; mode < count; ++mode) {
        const double scale = numeric::Norm(stiffness_vectors.Column(mode)) +
            std::abs(result.Eigenvalues[mode]) * numeric::Norm(mass_vectors.Column(mode));
        const double norm = numeric::Norm(residual.Column(mode));
        result.RelativeResiduals[mode] = scale == 0 ? norm : norm / scale;
        if (mode < certified_count && std::abs(result.Eigenvalues[mode]) > rigid_threshold)
            maximum_physical_residual = std::max(maximum_physical_residual, result.RelativeResiduals[mode]);
    }
    numeric::Matrix<double> gram = numeric::TransposeMultiply(result.Eigenvectors.View(), mass_vectors);
    for (uint32_t mode = 0; mode < count; ++mode) gram(mode, mode) -= 1;
    result.MassOrthogonalityError = numeric::Norm(gram.View());
    result.Converged = maximum_physical_residual <= residual_tolerance &&
        result.MassOrthogonalityError <= std::max(1e-9, 10 * residual_tolerance);
}

template<class ShiftInvert>
uint32_t RefineGeneralizedEigenpairs(
    const ShiftInvert &operation, const numeric::SparseMatrix &mass,
    const numeric::SparseMatrix &stiffness, double target_residual,
    double rigid_threshold, uint32_t certified_count, uint32_t max_iterations,
    numeric::Vector<double> &eigenvalues, numeric::Matrix<double> &eigenvectors,
    numeric::Matrix<double> &mass_vectors, numeric::Matrix<double> &stiffness_vectors,
    bool actions_initialized = false
) {
    constexpr double ClusterRelativeGap{1e-3};
    const uint32_t width = uint32_t(eigenvalues.size());
    uint32_t applications{};
    if (!actions_initialized) {
        mass_vectors = numeric::SymmetricMultiply(mass, eigenvectors.View());
        stiffness_vectors = numeric::SymmetricMultiply(stiffness, eigenvectors.View());
    }
    for (uint32_t iteration = 0; iteration < max_iterations; ++iteration) {
        const numeric::Matrix<double> residual = numeric::ColumnScaledDifference(stiffness_vectors.View(), mass_vectors.View(), eigenvalues.View());
        std::vector<bool> active(width);
        bool unconverged{};
        for (uint32_t mode = 0; mode < std::min(certified_count, width); ++mode) {
            if (std::abs(eigenvalues[mode]) <= rigid_threshold) continue;
            const double scale = numeric::Norm(stiffness_vectors.Column(mode)) +
                std::abs(eigenvalues[mode]) * numeric::Norm(mass_vectors.Column(mode));
            if (numeric::Norm(residual.Column(mode)) > target_residual * scale) {
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

        numeric::Matrix<double> active_residual(eigenvectors.rows(), active_modes.size());
        for (size_t column = 0; column < active_modes.size(); ++column)
            numeric::Copy(residual.Column(active_modes[column]), active_residual.Column(column));
        numeric::Matrix<double> correction(eigenvectors.rows(), active_modes.size());
        operation.solve_panel(active_residual.data(), correction.data(), int(active_modes.size()));
        applications += uint32_t(active_modes.size());

        numeric::Matrix<double> mass_correction = numeric::SymmetricMultiply(mass, correction.View());
        for (uint32_t pass = 0; pass < 2; ++pass) {
            const numeric::Matrix<double> coefficients = numeric::TransposeMultiply(eigenvectors.View(), mass_correction.View());
            numeric::SubtractProduct(correction.View(), eigenvectors.View(), coefficients.View());
            numeric::SubtractProduct(mass_correction.View(), mass_vectors.View(), coefficients.View());
        }
        numeric::Matrix<double> gram = numeric::TransposeMultiply(correction.View(), mass_correction.View());
        numeric::Symmetrize(gram.View());
        numeric::Vector<double> correction_values;
        if (!SelfAdjointEigenvectors(gram, correction_values)) break;
        const double threshold = correction_values[correction_values.size() - 1] * 1e-12;
        size_t first{};
        while (first < correction.cols() && correction_values[first] <= threshold) ++first;
        if (first == correction.cols()) break;
        const size_t correction_width = correction.cols() - first;
        numeric::Matrix<double> transform = numeric::Copy(gram.LastColumns(correction_width));
        numeric::Vector<double> inverse_sqrt(correction_width);
        for (size_t column = 0; column < correction_width; ++column)
            inverse_sqrt[column] = 1 / std::sqrt(correction_values[first + column]);
        numeric::ScaleColumns(transform.View(), inverse_sqrt.View());
        correction = numeric::Multiply(correction.View(), transform.View());
        mass_correction = numeric::Multiply(mass_correction.View(), transform.View());
        const numeric::Matrix<double> stiffness_correction = numeric::SymmetricMultiply(stiffness, correction.View());

        numeric::Matrix<double> space(eigenvectors.rows(), width + correction_width);
        numeric::Copy(eigenvectors.View(), space.FirstColumns(width));
        numeric::Copy(correction.View(), space.LastColumns(correction_width));
        numeric::Matrix<double> mass_space(eigenvectors.rows(), width + correction_width);
        numeric::Copy(mass_vectors.View(), mass_space.FirstColumns(width));
        numeric::Copy(mass_correction.View(), mass_space.LastColumns(correction_width));
        numeric::Matrix<double> stiffness_space(eigenvectors.rows(), width + correction_width);
        numeric::Copy(stiffness_vectors.View(), stiffness_space.FirstColumns(width));
        numeric::Copy(stiffness_correction.View(), stiffness_space.LastColumns(correction_width));
        numeric::Matrix<double> projected = numeric::TransposeMultiply(space.View(), stiffness_space.View());
        numeric::Symmetrize(projected.View());
        numeric::Vector<double> projected_values;
        if (!SelfAdjointEigenvectors(projected, projected_values)) break;
        const auto rotation = projected.FirstColumns(width);
        eigenvalues = numeric::Copy(projected_values.First(width));
        eigenvectors = numeric::Multiply(space.View(), rotation);
        mass_vectors = numeric::Multiply(mass_space.View(), rotation);
        stiffness_vectors = numeric::Multiply(stiffness_space.View(), rotation);
    }
    return applications;
}

template<class ShiftInvert>
GeneralizedEigenResult SolveGeneralizedInverseIteration(
    ShiftInvert &operation, const numeric::SparseMatrix &mass,
    const numeric::SparseMatrix &stiffness, const GeneralizedEigenOptions &options,
    const numeric::Matrix<float> *seed = nullptr, GeneralizedEigenControl control = {}
) {
    const uint32_t n = uint32_t(mass.rows());
    const uint32_t count = std::min(options.Count, n > 0 ? n - 1 : 0);
    const uint32_t subspace_size = std::min(std::max(options.SubspaceSize, count + 1), n);
    GeneralizedEigenResult result;
    if (!count || subspace_size <= count || stiffness.rows() != mass.rows() || stiffness.cols() != mass.cols()) return result;

    bool operation_ready{};
    if (seed && seed->rows() == n && seed->cols() >= count) {
        numeric::Matrix<double> seed_vectors = numeric::Cast<double>(seed->FirstColumns(count));
        numeric::Matrix<double> seed_mass = numeric::SymmetricMultiply(mass, seed_vectors.View());
        numeric::Matrix<double> seed_stiffness = numeric::SymmetricMultiply(stiffness, seed_vectors.View());
        numeric::Matrix<double> projected_mass = numeric::TransposeMultiply(seed_vectors.View(), seed_mass.View());
        numeric::Matrix<double> projected_stiffness = numeric::TransposeMultiply(seed_vectors.View(), seed_stiffness.View());
        numeric::Symmetrize(projected_mass.View());
        numeric::Symmetrize(projected_stiffness.View());
        numeric::Vector<double> seed_values;
        if (GeneralizedEigenvectors(projected_stiffness, projected_mass, seed_values)) {
            result.Eigenvalues = std::move(seed_values);
            result.Eigenvectors = numeric::Multiply(seed_vectors.View(), projected_stiffness.View());
            seed_mass = numeric::Multiply(seed_mass.View(), projected_stiffness.View());
            seed_stiffness = numeric::Multiply(seed_stiffness.View(), projected_stiffness.View());
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
    numeric::Matrix<double> mass_space(n, subspace_size);
    {
        numeric::Matrix<double> space(n, subspace_size);
        std::mt19937_64 random{options.RandomSeed};
        if (seeded) {
            const auto converted = numeric::Cast<double>(seed->FirstColumns(seeded));
            numeric::Copy(converted.View(), space.FirstColumns(seeded));
        }
        for (uint32_t column = seeded; column < subspace_size; ++column)
            for (uint32_t row = 0; row < n; ++row) space(row, column) = (random() & 1) ? 1.0 : -1.0;
        numeric::SymmetricMultiply(mass, space.View(), mass_space.View());
    }

    numeric::Matrix<double> locked_vectors(n, count), locked_mass_vectors(n, count);
    numeric::Vector<double> locked_shifted_values(count);
    numeric::Vector<double> previous_values(count, std::numeric_limits<double>::max());
    uint32_t locked{};
    for (uint32_t iteration = 0; iteration < options.MaxIterations; ++iteration) {
        if (control.Cancel && control.Cancel->load(std::memory_order_relaxed)) return {};
        const uint32_t width = subspace_size - locked;
        numeric::Matrix<double> space(n, width);
        operation.solve_panel(mass_space.data(), space.data(), int(width));
        result.OpApplications += width;

        numeric::Matrix<double> mass_vectors = numeric::SymmetricMultiply(mass, space.View());
        numeric::Matrix<double> projected_shifted = numeric::TransposeMultiply(space.View(), mass_space.View());
        numeric::Matrix<double> projected_mass = numeric::TransposeMultiply(space.View(), mass_vectors.View());
        if (locked) {
            const auto locked_vectors_view = locked_vectors.FirstColumns(locked);
            const auto locked_mass_view = locked_mass_vectors.FirstColumns(locked);
            const numeric::Matrix<double> coefficients = numeric::TransposeMultiply(locked_vectors_view, mass_vectors.View());
            numeric::SubtractProduct(space.View(), locked_vectors_view, coefficients.View());
            numeric::SubtractProduct(mass_vectors.View(), locked_mass_view, coefficients.View());
            numeric::Matrix<double> weighted = numeric::Copy(coefficients.View());
            ScaleRows(weighted.View(), locked_shifted_values.First(locked));
            const numeric::Matrix<double> correction = numeric::TransposeMultiply(coefficients.View(), weighted.View());
            Subtract(projected_shifted.View(), correction.View());
            projected_mass = numeric::TransposeMultiply(space.View(), mass_vectors.View());
        }
        numeric::Symmetrize(projected_shifted.View());
        numeric::Symmetrize(projected_mass.View());
        const numeric::Vector<double> inverse_norm = InverseSqrtDiagonal(projected_mass.View());
        if (!numeric::AllFinite(inverse_norm.View())) return {};
        numeric::ScaleRowsAndColumns(projected_shifted.View(), inverse_norm.View());
        numeric::ScaleRowsAndColumns(projected_mass.View(), inverse_norm.View());
        numeric::Vector<double> decomposition_values;
        if (!GeneralizedEigenvectors(projected_shifted, projected_mass, decomposition_values)) return {};
        numeric::Matrix<double> rotation = std::move(projected_shifted);
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
            numeric::Multiply(space.View(), rotation.FirstColumns(newly_locked), locked_vectors.ColumnsAt(locked, newly_locked));
            numeric::Multiply(mass_vectors.View(), rotation.FirstColumns(newly_locked), locked_mass_vectors.ColumnsAt(locked, newly_locked));
            numeric::Copy(decomposition_values.First(newly_locked), locked_shifted_values.Subvector(locked, newly_locked));
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
                const numeric::Matrix<double> locked_stiffness = numeric::SymmetricMultiply(stiffness, locked_vectors.View());
                CertifyGeneralizedEigenResult(result, locked_mass_vectors.View(), locked_stiffness.View(), certified_count, rigid_threshold, options.ResidualTolerance);
                if (result.Converged) return result;
            }
            const uint32_t guard_count = width - newly_locked;
            numeric::Vector<double> refined_values(count + guard_count);
            numeric::Copy(previous_values.View(), refined_values.First(count));
            numeric::Matrix<double> refined_vectors(n, count + guard_count);
            numeric::Copy(locked_vectors.View(), refined_vectors.FirstColumns(count));
            if (guard_count) {
                for (uint32_t guard = 0; guard < guard_count; ++guard)
                    refined_values[count + guard] = decomposition_values[decomposition_values.size() - guard_count + guard] + options.Shift;
                numeric::Multiply(space.View(), rotation.LastColumns(guard_count), refined_vectors.LastColumns(guard_count));
            }
            numeric::Matrix<double> refined_mass, refined_stiffness;
            result.OpApplications += RefineGeneralizedEigenpairs(
                operation, mass, stiffness, options.ResidualTolerance, rigid_threshold, count,
                options.MaxRefinementIterations, refined_values, refined_vectors, refined_mass, refined_stiffness
            );
            result.Eigenvalues = numeric::Copy(refined_values.First(count));
            result.Eigenvectors = numeric::Copy(refined_vectors.FirstColumns(count));
            CertifyGeneralizedEigenResult(
                result, refined_mass.FirstColumns(count), refined_stiffness.FirstColumns(count),
                certified_count, rigid_threshold, options.ResidualTolerance
            );
            return result;
        }
        const uint32_t retained = width - newly_locked;
        mass_space.Resize(n, retained);
        numeric::Multiply(mass_vectors.View(), rotation.LastColumns(retained), mass_space.View());
    }
    return {};
}

template<class ShiftInvert>
GeneralizedEigenResult SolveGeneralizedBlockKrylov(
    ShiftInvert &operation, const numeric::SparseMatrix &mass,
    const numeric::SparseMatrix &stiffness, const GeneralizedEigenOptions &options,
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
    numeric::Matrix<double> basis(n, capacity), mass_basis(n, capacity), action_basis(n, capacity);
    const auto append = [&](numeric::Matrix<double> vectors, numeric::Matrix<double> mass_vectors, uint32_t used) {
        if (block_width == 1) {
            for (uint32_t pass = 0; pass < 2 && used; ++pass) {
                const numeric::Matrix<double> coefficients = numeric::TransposeMultiply(basis.FirstColumns(used), mass_vectors.View());
                numeric::SubtractProduct(vectors.View(), basis.FirstColumns(used), coefficients.View());
                numeric::SubtractProduct(mass_vectors.View(), mass_basis.FirstColumns(used), coefficients.View());
            }
        } else {
            constexpr uint32_t OrthogonalizationBlock{16};
            for (uint32_t first = 0; first < used; first += OrthogonalizationBlock) {
                const uint32_t width = std::min(OrthogonalizationBlock, used - first);
                const numeric::Matrix<double> coefficients = numeric::TransposeMultiply(basis.ColumnsAt(first, width), mass_vectors.View());
                numeric::SubtractProduct(vectors.View(), basis.ColumnsAt(first, width), coefficients.View());
                numeric::SubtractProduct(mass_vectors.View(), mass_basis.ColumnsAt(first, width), coefficients.View());
            }
        }
        numeric::Matrix<double> gram = numeric::TransposeMultiply(vectors.View(), mass_vectors.View());
        const numeric::Vector<double> inverse_norm = InverseSqrtDiagonal(gram.View());
        if (!numeric::AllFinite(inverse_norm.View())) return uint32_t{};
        numeric::Symmetrize(gram.View());
        numeric::ScaleRowsAndColumns(gram.View(), inverse_norm.View());
        numeric::Vector<double> values;
        if (!SelfAdjointEigenvectors(gram, values)) return uint32_t{};
        double maximum{};
        for (double value : values.Values) maximum = std::max(maximum, std::abs(value));
        const double threshold = maximum * 1e-12;
        size_t first{};
        while (first < values.size() && values[first] <= threshold) ++first;
        const uint32_t rank = uint32_t(values.size() - first);
        const uint32_t retained = std::min(rank, capacity - used);
        if (!retained) return uint32_t{};
        numeric::Matrix<double> transform = numeric::Copy(gram.ColumnsAt(first, retained));
        ScaleRows(transform.View(), inverse_norm.View());
        numeric::Vector<double> inverse_sqrt(retained);
        for (uint32_t column = 0; column < retained; ++column) inverse_sqrt[column] = 1 / std::sqrt(values[first + column]);
        numeric::ScaleColumns(transform.View(), inverse_sqrt.View());
        numeric::Multiply(vectors.View(), transform.View(), basis.ColumnsAt(used, retained));
        numeric::Multiply(mass_vectors.View(), transform.View(), mass_basis.ColumnsAt(used, retained));
        return retained;
    };

    const uint32_t initial_width = std::min(block_width, capacity);
    numeric::Matrix<double> initial(n, initial_width);
    std::mt19937_64 random{options.RandomSeed};
    for (uint32_t column = 0; column < initial_width; ++column)
        for (uint32_t row = 0; row < n; ++row) initial(row, column) = (random() & 1) ? 1.0 : -1.0;
    numeric::Matrix<double> initial_mass = numeric::SymmetricMultiply(mass, initial.View());
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
                numeric::Matrix<double> next = numeric::Copy(action_basis.ColumnsAt(applied, width));
                numeric::Matrix<double> next_mass = numeric::SymmetricMultiply(mass, next.View());
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
    numeric::Vector<double> values;
    numeric::Matrix<double> vectors, mass_vectors, stiffness_vectors;
    const auto extract = [&](uint32_t width) {
        numeric::Matrix<double> projected = numeric::TransposeMultiply(mass_basis.FirstColumns(width), action_basis.FirstColumns(width));
        numeric::Symmetrize(projected.View());
        numeric::Matrix<double> projected_mass = numeric::TransposeMultiply(basis.FirstColumns(width), mass_basis.FirstColumns(width));
        numeric::Symmetrize(projected_mass.View());
        numeric::Vector<double> inverse_values;
        if (!GeneralizedEigenvectors(projected, projected_mass, inverse_values)) return false;
        numeric::Matrix<double> rotation(width, retained_size);
        values.Resize(retained_size);
        for (uint32_t mode = 0; mode < retained_size; ++mode) {
            const size_t index = inverse_values.size() - 1 - mode;
            const double inverse_value = inverse_values[index];
            if (!(inverse_value > 0) || !std::isfinite(inverse_value)) return false;
            values[mode] = options.Shift + 1 / inverse_value;
            numeric::Copy(projected.Column(index), rotation.Column(mode));
        }
        vectors.Resize(n, retained_size);
        mass_vectors.Resize(n, retained_size);
        numeric::Multiply(basis.FirstColumns(width), rotation.View(), vectors.View());
        numeric::Multiply(mass_basis.FirstColumns(width), rotation.View(), mass_vectors.View());
        stiffness_vectors = numeric::SymmetricMultiply(stiffness, vectors.View());
        result.Eigenvalues = numeric::Copy(values.First(count));
        result.Eigenvectors = numeric::Copy(vectors.FirstColumns(count));
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
    result.Eigenvalues = numeric::Copy(values.First(count));
    result.Eigenvectors = numeric::Copy(vectors.FirstColumns(count));
    CertifyGeneralizedEigenResult(result, mass_vectors.FirstColumns(count), stiffness_vectors.FirstColumns(count), certified_count, rigid_threshold, options.ResidualTolerance);
    if (control.Progress) control.Progress->store(control.ProgressEnd, std::memory_order_relaxed);
    return result;
}

template<class ShiftInvert>
GeneralizedEigenResult SolveGeneralizedEigenproblem(
    ShiftInvert &operation, const numeric::SparseMatrix &mass,
    const numeric::SparseMatrix &stiffness, const GeneralizedEigenOptions &options,
    const numeric::Matrix<float> *seed = nullptr, GeneralizedEigenControl control = {}
) {
    if (seed || options.Count <= 12 || options.Count >= 128 || mass.rows() >= 100000)
        return SolveGeneralizedInverseIteration(operation, mass, stiffness, options, seed, control);
    auto result = SolveGeneralizedBlockKrylov(operation, mass, stiffness, options, control);
    if (result.Converged) return result;
    return SolveGeneralizedInverseIteration(operation, mass, stiffness, options, seed, control);
}
} // namespace modal::eigensolver
