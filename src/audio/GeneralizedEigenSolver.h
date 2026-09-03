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
    Eigen::VectorXd Eigenvalues;
    Eigen::MatrixXd Eigenvectors;
    Eigen::VectorXd RelativeResiduals;
    double MassOrthogonalityError{};
    uint32_t Iterations{}, OpApplications{};
    bool Converged{};
};

inline void CertifyGeneralizedEigenResult(
    GeneralizedEigenResult &result, const Eigen::MatrixXd &mass_vectors,
    const Eigen::MatrixXd &stiffness_vectors, uint32_t certified_count,
    double rigid_threshold, double residual_tolerance
) {
    const uint32_t count = uint32_t(result.Eigenvalues.size());
    const Eigen::MatrixXd residual = stiffness_vectors - mass_vectors * result.Eigenvalues.asDiagonal();
    result.RelativeResiduals.resize(count);
    double maximum_physical_residual{};
    for (uint32_t mode = 0; mode < count; ++mode) {
        const double scale = stiffness_vectors.col(mode).norm() +
            std::abs(result.Eigenvalues[mode]) * mass_vectors.col(mode).norm();
        result.RelativeResiduals[mode] = scale == 0 ? residual.col(mode).norm() : residual.col(mode).norm() / scale;
        if (mode < certified_count && std::abs(result.Eigenvalues[mode]) > rigid_threshold)
            maximum_physical_residual = std::max(maximum_physical_residual, result.RelativeResiduals[mode]);
    }
    result.MassOrthogonalityError =
        (result.Eigenvectors.transpose() * mass_vectors - Eigen::MatrixXd::Identity(count, count)).norm();
    result.Converged = maximum_physical_residual <= residual_tolerance &&
        result.MassOrthogonalityError <= std::max(1e-9, 10 * residual_tolerance);
}

template<class ShiftInvert>
uint32_t RefineGeneralizedEigenpairs(
    const ShiftInvert &operation, const Eigen::SparseMatrix<double> &mass,
    const Eigen::SparseMatrix<double> &stiffness, double target_residual,
    double rigid_threshold, uint32_t certified_count, uint32_t max_iterations,
    Eigen::VectorXd &eigenvalues, Eigen::MatrixXd &eigenvectors,
    Eigen::MatrixXd &mass_vectors, Eigen::MatrixXd &stiffness_vectors,
    bool actions_initialized = false
) {
    constexpr double ClusterRelativeGap{1e-3};
    const auto mass_operator = mass.selfadjointView<Eigen::Lower>();
    const auto stiffness_operator = stiffness.selfadjointView<Eigen::Lower>();
    const uint32_t width = uint32_t(eigenvalues.size());
    uint32_t applications{};
    if (!actions_initialized) {
        mass_vectors.noalias() = mass_operator * eigenvectors;
        stiffness_vectors.noalias() = stiffness_operator * eigenvectors;
    }
    for (uint32_t iteration = 0; iteration < max_iterations; ++iteration) {
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
        mass_correction *= transform;
        const Eigen::MatrixXd stiffness_correction = stiffness_operator * correction;

        Eigen::MatrixXd space(eigenvectors.rows(), width + correction_width);
        space << eigenvectors, correction;
        Eigen::MatrixXd mass_space(eigenvectors.rows(), width + correction_width);
        mass_space << mass_vectors, mass_correction;
        Eigen::MatrixXd stiffness_space(eigenvectors.rows(), width + correction_width);
        stiffness_space << stiffness_vectors, stiffness_correction;
        Eigen::MatrixXd projected = space.transpose() * stiffness_space;
        projected = (0.5 * (projected + projected.transpose())).eval();
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomposition{projected};
        if (decomposition.info() != Eigen::Success) break;
        const auto rotation = decomposition.eigenvectors().leftCols(width);
        eigenvalues = decomposition.eigenvalues().head(width);
        eigenvectors = space * rotation;
        mass_vectors = mass_space * rotation;
        stiffness_vectors = stiffness_space * rotation;
    }
    return applications;
}

template<class ShiftInvert>
GeneralizedEigenResult SolveGeneralizedInverseIteration(
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

    const auto mass_operator = mass.selfadjointView<Eigen::Lower>();
    bool operation_ready{};
    if (seed && seed->rows() == n && seed->cols() >= count) {
        Eigen::MatrixXd seed_vectors = seed->leftCols(count).template cast<double>();
        Eigen::MatrixXd seed_mass = mass_operator * seed_vectors;
        Eigen::MatrixXd seed_stiffness = stiffness.selfadjointView<Eigen::Lower>() * seed_vectors;
        Eigen::MatrixXd projected_mass = seed_vectors.transpose() * seed_mass;
        Eigen::MatrixXd projected_stiffness = seed_vectors.transpose() * seed_stiffness;
        projected_mass = (0.5 * (projected_mass + projected_mass.transpose())).eval();
        projected_stiffness = (0.5 * (projected_stiffness + projected_stiffness.transpose())).eval();
        const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> decomposition{
            projected_stiffness,
            projected_mass,
        };
        if (decomposition.info() == Eigen::Success) {
            result.Eigenvalues = decomposition.eigenvalues();
            result.Eigenvectors = seed_vectors * decomposition.eigenvectors();
            seed_mass = seed_mass * decomposition.eigenvectors();
            seed_stiffness = seed_stiffness * decomposition.eigenvectors();
            const uint32_t certified_count = options.CertifiedCount ?
                std::min(options.CertifiedCount, count) :
                count;
            const double rigid_threshold = std::max(std::abs(options.Shift) * 1e-4, 1e-12);
            CertifyGeneralizedEigenResult(
                result, seed_mass, seed_stiffness, certified_count,
                rigid_threshold, options.ResidualTolerance
            );
            if (result.Converged) return result;
            operation.set_shift(options.Shift);
            operation_ready = true;
            result.OpApplications += RefineGeneralizedEigenpairs(
                operation, mass, stiffness, options.ResidualTolerance, rigid_threshold,
                certified_count, std::min(2u, options.MaxRefinementIterations),
                result.Eigenvalues, result.Eigenvectors, seed_mass, seed_stiffness, true
            );
            CertifyGeneralizedEigenResult(
                result, seed_mass, seed_stiffness, certified_count,
                rigid_threshold, options.ResidualTolerance
            );
            if (result.Converged) return result;
            result = {};
        }
    }

    if (!operation_ready) operation.set_shift(options.Shift);
    const uint32_t seeded = seed && seed->rows() == n ? std::min(uint32_t(seed->cols()), subspace_size) : 0;
    Eigen::MatrixXd mass_space(n, subspace_size);
    {
        Eigen::MatrixXd space(n, subspace_size);
        std::mt19937_64 random{options.RandomSeed};
        if (seeded) space.leftCols(seeded) = seed->leftCols(seeded).template cast<double>();
        for (uint32_t column = seeded; column < subspace_size; ++column)
            for (uint32_t row = 0; row < n; ++row) space(row, column) = (random() & 1) ? 1.0 : -1.0;
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
            const double rigid_threshold = std::max(std::abs(options.Shift) * 1e-4, 1e-12);
            const uint32_t certified_count = options.CertifiedCount ? std::min(options.CertifiedCount, count) : count;
            if (seeded) {
                result.Eigenvalues = previous_values;
                result.Eigenvectors = locked_vectors;
                const Eigen::MatrixXd locked_stiffness =
                    stiffness.selfadjointView<Eigen::Lower>() * locked_vectors;
                CertifyGeneralizedEigenResult(
                    result, locked_mass_vectors, locked_stiffness, certified_count,
                    rigid_threshold, options.ResidualTolerance
                );
                if (result.Converged) return result;
            }
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
            Eigen::MatrixXd refined_mass(n, count + guard_count), refined_stiffness(n, count + guard_count);
            result.OpApplications += RefineGeneralizedEigenpairs(
                operation, mass, stiffness, options.ResidualTolerance, rigid_threshold, count,
                options.MaxRefinementIterations, refined_values, refined_vectors, refined_mass, refined_stiffness
            );
            result.Eigenvalues = refined_values.head(count);
            result.Eigenvectors = refined_vectors.leftCols(count);
            const auto final_mass = refined_mass.leftCols(count);
            const auto final_stiffness = refined_stiffness.leftCols(count);
            CertifyGeneralizedEigenResult(
                result, final_mass, final_stiffness, certified_count,
                rigid_threshold, options.ResidualTolerance
            );
            return result;
        }
        mass_space.noalias() = mass_vectors * rotation.rightCols(width - newly_locked);
    }
    return {};
}

template<class ShiftInvert>
GeneralizedEigenResult SolveGeneralizedBlockKrylov(
    ShiftInvert &operation, const Eigen::SparseMatrix<double> &mass,
    const Eigen::SparseMatrix<double> &stiffness, const GeneralizedEigenOptions &options,
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
    if (!count || subspace_size <= count || stiffness.rows() != mass.rows() || stiffness.cols() != mass.cols())
        return result;

    operation.set_shift(options.Shift);
    const auto mass_operator = mass.selfadjointView<Eigen::Lower>();
    Eigen::MatrixXd basis(n, capacity), mass_basis(n, capacity), action_basis(n, capacity);
    const auto append = [&](Eigen::MatrixXd vectors, Eigen::MatrixXd mass_vectors, uint32_t used) {
        if (block_width == 1) {
            for (uint32_t pass = 0; pass < 2 && used; ++pass) {
                const Eigen::MatrixXd coefficients = basis.leftCols(used).transpose() * mass_vectors;
                vectors.noalias() -= basis.leftCols(used) * coefficients;
                mass_vectors.noalias() -= mass_basis.leftCols(used) * coefficients;
            }
        } else {
            constexpr uint32_t OrthogonalizationBlock{16};
            for (uint32_t first = 0; first < used; first += OrthogonalizationBlock) {
                const uint32_t width = std::min(OrthogonalizationBlock, used - first);
                const Eigen::MatrixXd coefficients = basis.middleCols(first, width).transpose() * mass_vectors;
                vectors.noalias() -= basis.middleCols(first, width) * coefficients;
                mass_vectors.noalias() -= mass_basis.middleCols(first, width) * coefficients;
            }
        }
        Eigen::MatrixXd gram = vectors.transpose() * mass_vectors;
        const Eigen::VectorXd inverse_norm = gram.diagonal().cwiseSqrt().cwiseInverse();
        if (!inverse_norm.allFinite()) return uint32_t{};
        gram = inverse_norm.asDiagonal() * (0.5 * (gram + gram.transpose())).eval() * inverse_norm.asDiagonal();
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomposition{gram};
        if (decomposition.info() != Eigen::Success) return uint32_t{};
        const double threshold = decomposition.eigenvalues().cwiseAbs().maxCoeff() * 1e-12;
        Eigen::Index first{};
        while (first < decomposition.eigenvalues().size() && decomposition.eigenvalues()[first] <= threshold) ++first;
        const uint32_t rank = uint32_t(decomposition.eigenvalues().size() - first);
        const uint32_t retained = std::min(rank, capacity - used);
        if (!retained) return uint32_t{};
        const Eigen::MatrixXd transform = inverse_norm.asDiagonal() *
            decomposition.eigenvectors().rightCols(rank).leftCols(retained) *
            decomposition.eigenvalues().tail(rank).head(retained).cwiseSqrt().cwiseInverse().asDiagonal();
        basis.middleCols(used, retained).noalias() = vectors * transform;
        mass_basis.middleCols(used, retained).noalias() = mass_vectors * transform;
        return retained;
    };

    const uint32_t initial_width = std::min(block_width, capacity);
    Eigen::MatrixXd initial(n, initial_width);
    std::mt19937_64 random{options.RandomSeed};
    for (uint32_t column = 0; column < initial_width; ++column)
        for (uint32_t row = 0; row < n; ++row) initial(row, column) = (random() & 1) ? 1.0 : -1.0;
    Eigen::MatrixXd initial_mass = mass_operator * initial;
    uint32_t used = append(std::move(initial), std::move(initial_mass), 0);
    uint32_t applied{};
    const double rigid_threshold = std::max(std::abs(options.Shift) * 1e-4, 1e-12);
    const uint32_t certified_count = options.CertifiedCount ? std::min(options.CertifiedCount, count) : count;
    const auto grow = [&](uint32_t target) {
        while (applied < target) {
            if (control.Cancel && control.Cancel->load(std::memory_order_relaxed)) return false;
            if (applied >= used) return false;
            const uint32_t width = std::min(block_width, used - applied);
            operation.solve_panel(mass_basis.col(applied).data(), action_basis.col(applied).data(), int(width));
            result.OpApplications += width;
            if (used < capacity) {
                Eigen::MatrixXd next = action_basis.middleCols(applied, width);
                Eigen::MatrixXd next_mass = mass_operator * next;
                const uint32_t added = append(std::move(next), std::move(next_mass), used);
                if (!added) return false;
                used += added;
            }
            applied += width;
            ++result.Iterations;
            if (control.Progress) {
                const float fraction = float(applied) / float(capacity);
                control.Progress->store(
                    control.ProgressBegin + fraction * (control.ProgressEnd - control.ProgressBegin),
                    std::memory_order_relaxed
                );
            }
        }
        return true;
    };
    Eigen::VectorXd values;
    Eigen::MatrixXd vectors, mass_vectors, stiffness_vectors;
    const auto extract = [&](uint32_t width) {
        Eigen::MatrixXd projected = mass_basis.leftCols(width).transpose() * action_basis.leftCols(width);
        projected = (0.5 * (projected + projected.transpose())).eval();
        Eigen::MatrixXd projected_mass = basis.leftCols(width).transpose() * mass_basis.leftCols(width);
        projected_mass = (0.5 * (projected_mass + projected_mass.transpose())).eval();
        const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> decomposition{projected, projected_mass};
        if (decomposition.info() != Eigen::Success) return false;
        Eigen::MatrixXd rotation(width, retained_size);
        values.resize(retained_size);
        for (uint32_t mode = 0; mode < retained_size; ++mode) {
            const Eigen::Index index = decomposition.eigenvalues().size() - 1 - mode;
            const double inverse_value = decomposition.eigenvalues()[index];
            if (!(inverse_value > 0) || !std::isfinite(inverse_value)) return false;
            values[mode] = options.Shift + 1 / inverse_value;
            rotation.col(mode) = decomposition.eigenvectors().col(index);
        }
        vectors.noalias() = basis.leftCols(width) * rotation;
        mass_vectors.noalias() = mass_basis.leftCols(width) * rotation;
        stiffness_vectors.noalias() = stiffness.selfadjointView<Eigen::Lower>() * vectors;
        result.Eigenvalues = values.head(count);
        result.Eigenvectors = vectors.leftCols(count);
        CertifyGeneralizedEigenResult(
            result, mass_vectors.leftCols(count), stiffness_vectors.leftCols(count),
            certified_count, rigid_threshold, options.ResidualTolerance
        );
        return true;
    };
    if (!grow(subspace_size) || !extract(subspace_size)) return {};
    double maximum_residual{};
    for (uint32_t mode = 0; mode < certified_count; ++mode)
        if (std::abs(values[mode]) > rigid_threshold)
            maximum_residual = std::max(maximum_residual, result.RelativeResiduals[mode]);
    if (capacity > subspace_size && maximum_residual > options.ExtensionResidual) {
        if (!grow(capacity) || !extract(capacity)) return {};
    }
    result.OpApplications += RefineGeneralizedEigenpairs(
        operation, mass, stiffness, options.ResidualTolerance, rigid_threshold, count,
        options.MaxRefinementIterations, values, vectors, mass_vectors, stiffness_vectors, block_width > 1
    );
    result.Eigenvalues = values.head(count);
    result.Eigenvectors = vectors.leftCols(count);
    CertifyGeneralizedEigenResult(
        result, mass_vectors.leftCols(count), stiffness_vectors.leftCols(count),
        certified_count, rigid_threshold, options.ResidualTolerance
    );
    if (control.Progress) control.Progress->store(control.ProgressEnd, std::memory_order_relaxed);
    return result;
}

template<class ShiftInvert>
GeneralizedEigenResult SolveGeneralizedEigenproblem(
    ShiftInvert &operation, const Eigen::SparseMatrix<double> &mass,
    const Eigen::SparseMatrix<double> &stiffness, const GeneralizedEigenOptions &options,
    const Eigen::MatrixXf *seed = nullptr, GeneralizedEigenControl control = {}
) {
    if (seed || options.Count <= 12 || options.Count >= 128 || mass.rows() >= 100000)
        return SolveGeneralizedInverseIteration(operation, mass, stiffness, options, seed, control);
    auto result = SolveGeneralizedBlockKrylov(operation, mass, stiffness, options, control);
    if (result.Converged) return result;
    return SolveGeneralizedInverseIteration(operation, mass, stiffness, options, seed, control);
}
} // namespace modal::detail
