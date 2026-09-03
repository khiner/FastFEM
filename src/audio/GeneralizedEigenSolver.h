#pragma once

#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace modal::detail {
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
};

struct GeneralizedEigenControl {
    const std::atomic<bool> *Cancel{};
    std::atomic<float> *Progress{};
    float ProgressBegin{};
    float ProgressEnd{1};
};

struct GeneralizedEigenResult {
    Eigen::VectorXd Eigenvalues;
    Eigen::MatrixXd Eigenvectors;
    Eigen::VectorXd RelativeResiduals;
    double MassOrthogonalityError{};
    uint32_t Iterations{}, OpApplications{};
    bool Converged{};
};

template<class ShiftInvert>
uint32_t RefineGeneralizedEigenpairs(
    const ShiftInvert &operation, const Eigen::SparseMatrix<double> &mass,
    const Eigen::SparseMatrix<double> &stiffness, double target_residual,
    double rigid_threshold, uint32_t certified_count, uint32_t max_iterations,
    Eigen::VectorXd &eigenvalues, Eigen::MatrixXd &eigenvectors
) {
    constexpr double ClusterRelativeGap{1e-3};
    const auto mass_operator = mass.selfadjointView<Eigen::Lower>();
    const auto stiffness_operator = stiffness.selfadjointView<Eigen::Lower>();
    const uint32_t width = uint32_t(eigenvalues.size());
    uint32_t applications{};
    for (uint32_t iteration = 0; iteration < max_iterations; ++iteration) {
        Eigen::MatrixXd mass_vectors = mass_operator * eigenvectors;
        Eigen::MatrixXd stiffness_vectors = stiffness_operator * eigenvectors;
        const Eigen::MatrixXd residual = stiffness_vectors - mass_vectors * eigenvalues.asDiagonal();
        std::vector<bool> active(width);
        bool unconverged{};
        for (Eigen::Index mode = 0; mode < std::min<Eigen::Index>(certified_count, eigenvalues.size()); ++mode) {
            if (std::abs(eigenvalues[mode]) <= rigid_threshold) continue;
            const double scale = stiffness_vectors.col(mode).norm() +
                std::abs(eigenvalues[mode]) * mass_vectors.col(mode).norm();
            if (residual.col(mode).norm() > target_residual * scale) {
                unconverged = true;
                active[size_t(mode)] = true;
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
                const double scale = std::max({
                    std::abs(eigenvalues[mode]),
                    std::abs(eigenvalues[mode + 1]),
                    rigid_threshold,
                });
                if (std::abs(eigenvalues[mode + 1] - eigenvalues[mode]) <= ClusterRelativeGap * scale) {
                    active[mode] = active[mode + 1] = true;
                    expanded = true;
                }
            }
        } while (expanded);
        std::vector<Eigen::Index> active_modes;
        active_modes.reserve(width);
        for (uint32_t mode = 0; mode < width; ++mode)
            if (active[mode]) active_modes.push_back(mode);
        if (active_modes.empty()) break;

        Eigen::MatrixXd active_residual(eigenvectors.rows(), Eigen::Index(active_modes.size()));
        for (size_t column = 0; column < active_modes.size(); ++column)
            active_residual.col(Eigen::Index(column)) = residual.col(active_modes[column]);
        Eigen::MatrixXd correction(eigenvectors.rows(), Eigen::Index(active_modes.size()));
        operation.solve_panel(active_residual.data(), correction.data(), int(active_modes.size()));
        applications += uint32_t(active_modes.size());

        Eigen::MatrixXd mass_correction = mass_operator * correction;
        for (uint32_t pass = 0; pass < 2; ++pass) {
            const Eigen::MatrixXd coefficients = eigenvectors.transpose() * mass_correction;
            correction.noalias() -= eigenvectors * coefficients;
            mass_correction.noalias() -= mass_vectors * coefficients;
        }
        Eigen::MatrixXd gram = correction.transpose() * mass_correction;
        gram = (0.5 * (gram + gram.transpose())).eval();
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> correction_decomposition{gram};
        if (correction_decomposition.info() != Eigen::Success) break;
        const double threshold = correction_decomposition.eigenvalues().maxCoeff() * 1e-12;
        Eigen::Index first{};
        while (first < correction.cols() && correction_decomposition.eigenvalues()[first] <= threshold) ++first;
        if (first == correction.cols()) break;
        const Eigen::Index correction_width = correction.cols() - first;
        const Eigen::MatrixXd transform = correction_decomposition.eigenvectors().rightCols(correction_width) *
            correction_decomposition.eigenvalues().tail(correction_width).cwiseSqrt().cwiseInverse().asDiagonal();
        correction *= transform;

        Eigen::MatrixXd space(eigenvectors.rows(), width + correction_width);
        space << eigenvectors, correction;
        Eigen::MatrixXd projected = space.transpose() * (stiffness_operator * space);
        projected = (0.5 * (projected + projected.transpose())).eval();
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomposition{projected};
        if (decomposition.info() != Eigen::Success) break;
        eigenvalues = decomposition.eigenvalues().head(width);
        eigenvectors = space * decomposition.eigenvectors().leftCols(width);
    }
    return applications;
}

template<class ShiftInvert>
GeneralizedEigenResult SolveGeneralizedEigenproblem(
    ShiftInvert &operation, const Eigen::SparseMatrix<double> &mass,
    const Eigen::SparseMatrix<double> &stiffness, const GeneralizedEigenOptions &options,
    const Eigen::MatrixXf *seed = nullptr, GeneralizedEigenControl control = {}
) {
    const uint32_t n = uint32_t(mass.rows());
    const uint32_t count = std::min(options.Count, n > 0 ? n - 1 : 0);
    const uint32_t subspace_size = std::min(std::max(options.SubspaceSize, count + 1), n);
    GeneralizedEigenResult result;
    if (!count || subspace_size <= count || stiffness.rows() != mass.rows() ||
        stiffness.cols() != mass.cols())
        return result;

    operation.set_shift(options.Shift);
    const auto mass_operator = mass.selfadjointView<Eigen::Lower>();
    Eigen::MatrixXd mass_space(n, subspace_size);
    {
        Eigen::MatrixXd space(n, subspace_size);
        std::mt19937_64 random{options.RandomSeed};
        std::normal_distribution<double> gaussian;
        const uint32_t seeded = seed && seed->rows() == n ?
            std::min(uint32_t(seed->cols()), subspace_size) :
            0;
        if (seeded) space.leftCols(seeded) = seed->leftCols(seeded).template cast<double>();
        for (uint32_t column = seeded; column < subspace_size; ++column)
            for (uint32_t row = 0; row < n; ++row) space(row, column) = gaussian(random);
        mass_space.noalias() = mass_operator * space;
    }

    Eigen::MatrixXd locked_vectors(n, count), locked_mass_vectors(n, count);
    Eigen::VectorXd locked_shifted_values(count);
    Eigen::VectorXd previous_values = Eigen::VectorXd::Constant(count, std::numeric_limits<double>::max());
    uint32_t locked{};
    for (uint32_t iteration = 0; iteration < options.MaxIterations; ++iteration) {
        if (control.Cancel && control.Cancel->load(std::memory_order_relaxed)) return {};
        const uint32_t width = subspace_size - locked;
        Eigen::MatrixXd space(n, width);
        operation.solve_panel(mass_space.data(), space.data(), int(width));
        result.OpApplications += width;

        Eigen::MatrixXd mass_vectors = mass_operator * space;
        Eigen::MatrixXd projected_shifted = space.transpose() * mass_space;
        Eigen::MatrixXd projected_mass = space.transpose() * mass_vectors;
        if (locked) {
            const Eigen::MatrixXd coefficients = locked_vectors.leftCols(locked).transpose() * mass_vectors;
            space.noalias() -= locked_vectors.leftCols(locked) * coefficients;
            mass_vectors.noalias() -= locked_mass_vectors.leftCols(locked) * coefficients;
            projected_shifted.noalias() -= coefficients.transpose() *
                locked_shifted_values.head(locked).asDiagonal() * coefficients;
            projected_mass = space.transpose() * mass_vectors;
        }
        projected_shifted = (0.5 * (projected_shifted + projected_shifted.transpose())).eval();
        projected_mass = (0.5 * (projected_mass + projected_mass.transpose())).eval();
        const Eigen::VectorXd inverse_norm = projected_mass.diagonal().cwiseSqrt().cwiseInverse();
        if (!inverse_norm.allFinite()) return {};
        projected_shifted = inverse_norm.asDiagonal() * projected_shifted * inverse_norm.asDiagonal();
        projected_mass = inverse_norm.asDiagonal() * projected_mass * inverse_norm.asDiagonal();
        const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> decomposition{
            projected_shifted,
            projected_mass,
        };
        if (decomposition.info() != Eigen::Success) return {};
        const Eigen::MatrixXd rotation = inverse_norm.asDiagonal() * decomposition.eigenvectors();

        uint32_t newly_locked{};
        for (uint32_t mode = 0; mode < width && locked + mode < count; ++mode) {
            const double value = decomposition.eigenvalues()[mode] + options.Shift;
            const double relative_change = std::abs(value - previous_values[locked + mode]) /
                std::max(std::abs(value), std::abs(options.Shift));
            previous_values[locked + mode] = value;
            if (newly_locked == mode && relative_change < options.IterationTolerance) ++newly_locked;
        }
        while (newly_locked && locked + newly_locked < count && newly_locked < width) {
            const double left = decomposition.eigenvalues()[newly_locked - 1] + options.Shift;
            const double right = decomposition.eigenvalues()[newly_locked] + options.Shift;
            const double scale = std::max({std::abs(left), std::abs(right), std::abs(options.Shift) * 1e-4});
            if (std::abs(right - left) > 1e-3 * scale) break;
            --newly_locked;
        }
        if (newly_locked) {
            locked_vectors.middleCols(locked, newly_locked).noalias() =
                space * rotation.leftCols(newly_locked);
            locked_mass_vectors.middleCols(locked, newly_locked).noalias() =
                mass_vectors * rotation.leftCols(newly_locked);
            locked_shifted_values.segment(locked, newly_locked) =
                decomposition.eigenvalues().head(newly_locked);
            locked += newly_locked;
        }
        result.Iterations = iteration + 1;
        if (control.Progress) {
            const float fraction = float(locked) / float(count);
            control.Progress->store(
                control.ProgressBegin + fraction * (control.ProgressEnd - control.ProgressBegin),
                std::memory_order_relaxed
            );
        }
        if (locked == count) {
            const uint32_t guard_count = width - newly_locked;
            Eigen::VectorXd refined_values(count + guard_count);
            refined_values.head(count) = previous_values;
            Eigen::MatrixXd refined_vectors(n, count + guard_count);
            refined_vectors.leftCols(count) = locked_vectors;
            if (guard_count) {
                refined_values.tail(guard_count) =
                    decomposition.eigenvalues().tail(guard_count).array() + options.Shift;
                refined_vectors.rightCols(guard_count).noalias() =
                    space * rotation.rightCols(guard_count);
            }
            const double rigid_threshold = std::max(std::abs(options.Shift) * 1e-4, 1e-12);
            result.OpApplications += RefineGeneralizedEigenpairs(
                operation, mass, stiffness, options.ResidualTolerance, rigid_threshold, count,
                options.MaxRefinementIterations, refined_values, refined_vectors
            );
            result.Eigenvalues = refined_values.head(count);
            result.Eigenvectors = refined_vectors.leftCols(count);
            const Eigen::MatrixXd final_mass = mass_operator * result.Eigenvectors;
            const Eigen::MatrixXd final_stiffness =
                stiffness.selfadjointView<Eigen::Lower>() * result.Eigenvectors;
            const Eigen::MatrixXd residual =
                final_stiffness - final_mass * result.Eigenvalues.asDiagonal();
            result.RelativeResiduals.resize(count);
            double maximum_physical_residual{};
            const uint32_t certified_count = options.CertifiedCount ?
                std::min(options.CertifiedCount, count) :
                count;
            for (uint32_t mode = 0; mode < count; ++mode) {
                const double scale = final_stiffness.col(mode).norm() +
                    std::abs(result.Eigenvalues[mode]) * final_mass.col(mode).norm();
                result.RelativeResiduals[mode] = scale == 0 ? residual.col(mode).norm() :
                                                              residual.col(mode).norm() / scale;
                if (mode < certified_count && std::abs(result.Eigenvalues[mode]) > rigid_threshold)
                    maximum_physical_residual = std::max(maximum_physical_residual, result.RelativeResiduals[mode]);
            }
            result.MassOrthogonalityError =
                (result.Eigenvectors.transpose() * final_mass - Eigen::MatrixXd::Identity(count, count)).norm();
            result.Converged = maximum_physical_residual <= options.ResidualTolerance &&
                result.MassOrthogonalityError <= std::max(1e-9, 10 * options.ResidualTolerance);
            return result;
        }
        mass_space.noalias() = mass_vectors * rotation.rightCols(width - newly_locked);
    }
    return {};
}
} // namespace modal::detail
