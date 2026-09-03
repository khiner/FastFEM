#define EIGEN_USE_BLAS

#include "FiniteCellBlockEigensolver.h"

#include "CholeskyShiftInvert.h"
#include "FiniteCellMetal.h"
#include "FiniteCellOracle.h"
#include "SparseCholesky.h"
#include "numeric/Accelerate.h"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymGEigsShiftSolver.h>

#include <dispatch/dispatch.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr uint32_t LocalDofs{3 * modal::FiniteCellOperator::NodesPerCell};
constexpr size_t PackedLocalValues{size_t(LocalDofs) * (LocalDofs + 1) / 2};

double SecondsSince(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void RemoveLeadingColumns(Eigen::MatrixXd &matrix, uint32_t count) {
    const Eigen::Index remaining = matrix.cols() - count;
    std::memmove(matrix.data(), matrix.data() + matrix.rows() * count, size_t(matrix.rows() * remaining) * sizeof(double));
    matrix.conservativeResize(Eigen::NoChange, remaining);
}

void RemoveLeadingEntries(Eigen::VectorXd &vector, uint32_t count) {
    const Eigen::Index remaining = vector.size() - count;
    std::memmove(vector.data(), vector.data() + count, size_t(remaining) * sizeof(double));
    vector.conservativeResize(remaining);
}

struct Actions {
    const modal::FiniteCellOperator &Fem;
    std::optional<modal::FiniteCellOperator::PackedCutOperators> PackedCut;
    double Seconds{};

    void ApplyMass(const double *input, double *output, uint32_t width) {
        Measure([&] {
            if (PackedCut) Fem.ApplyMassShiftedExpandedPackedCut(*PackedCut, input, output, nullptr, width);
            else Fem.ApplyMass(input, output, width);
        });
    }

    void ApplyShifted(const double *input, double *output, uint32_t width, double alpha) {
        Measure([&] {
            if (PackedCut) Fem.ApplyMassShiftedExpandedPackedCut(*PackedCut, input, nullptr, output, width);
            else Fem.ApplyShifted(input, output, width, alpha);
        });
    }

    void ApplyMassShifted(
        const double *input, double *mass_output, double *shifted_output, uint32_t width, double alpha
    ) {
        Measure([&] {
            if (PackedCut) Fem.ApplyMassShiftedExpandedPackedCut(*PackedCut, input, mass_output, shifted_output, width);
            else Fem.ApplyMassShifted(input, mass_output, shifted_output, width, alpha);
        });
    }

    void Measure(auto &&operation) {
        const auto start = Clock::now();
        operation();
        Seconds += SecondsSince(start);
    }
};

Eigen::MatrixXd SymmetricCrossGram(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
    if (a.rows() != b.rows() || a.cols() != b.cols())
        throw std::invalid_argument("Symmetric cross-Gram operands must have equal dimensions.");
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(a.cols(), a.cols());
    numeric::SymmetricCrossGram(a.data(), b.data(), result.data(), uint32_t(a.rows()), uint32_t(a.cols()));
    return result;
}

void RotateTallTriple(
    const Eigen::MatrixXd &a, const Eigen::MatrixXd &b, const Eigen::MatrixXd &c,
    const Eigen::MatrixXd &rotation, Eigen::MatrixXd &a_rotated, Eigen::MatrixXd &b_rotated,
    Eigen::MatrixXd &c_rotated
) {
    a_rotated.resize(a.rows(), rotation.cols());
    b_rotated.resize(b.rows(), rotation.cols());
    c_rotated.resize(c.rows(), rotation.cols());
    auto b_future = std::async(std::launch::async, [&] { b_rotated.noalias() = b * rotation; });
    auto c_future = std::async(std::launch::async, [&] { c_rotated.noalias() = c * rotation; });
    a_rotated.noalias() = a * rotation;
    b_future.get();
    c_future.get();
}

// Replaces `vectors` with an M-orthonormal basis and returns false when fewer than `required` independent directions remain.
bool MOrthonormalize(Actions &actions, Eigen::MatrixXd &vectors, Eigen::MatrixXd &mass_vectors, uint32_t required) {
    mass_vectors.resize(actions.Fem.Dofs(), vectors.cols());
    actions.ApplyMass(vectors.data(), mass_vectors.data(), uint32_t(vectors.cols()));
    const Eigen::VectorXd inverse_norm = (vectors.cwiseProduct(mass_vectors).colwise().sum().array().sqrt().inverse()).matrix();
    if (!inverse_norm.allFinite()) return false;
    vectors *= inverse_norm.asDiagonal();
    mass_vectors *= inverse_norm.asDiagonal();
    Eigen::MatrixXd gram = vectors.transpose() * mass_vectors;
    gram = (0.5 * (gram + gram.transpose())).eval();
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomposition{gram};
    if (decomposition.info() != Eigen::Success) return false;
    const double threshold = decomposition.eigenvalues().cwiseAbs().maxCoeff() * 1e-12;
    Eigen::Index first{};
    while (first < decomposition.eigenvalues().size() && decomposition.eigenvalues()[first] <= threshold) ++first;
    if (decomposition.eigenvalues().size() - first < required) return false;
    const Eigen::MatrixXd transform = decomposition.eigenvectors().rightCols(decomposition.eigenvalues().size() - first) *
        decomposition.eigenvalues().tail(decomposition.eigenvalues().size() - first).cwiseSqrt().cwiseInverse().asDiagonal();
    vectors = vectors * transform;
    mass_vectors = mass_vectors * transform;
    return true;
}

bool Ritz(
    Actions &actions, double alpha, Eigen::MatrixXd &space, Eigen::MatrixXd &mass_space, uint32_t count,
    Eigen::MatrixXd &vectors, Eigen::MatrixXd &mass_vectors, Eigen::MatrixXd &shifted_vectors,
    Eigen::VectorXd &values, double &seconds
) {
    const auto start = Clock::now();
    const double action_start = actions.Seconds;
    if (!MOrthonormalize(actions, space, mass_space, count)) return false;
    Eigen::MatrixXd shifted_space(actions.Fem.Dofs(), space.cols());
    actions.ApplyShifted(space.data(), shifted_space.data(), uint32_t(space.cols()), alpha);
    Eigen::MatrixXd projected = space.transpose() * shifted_space;
    projected = (0.5 * (projected + projected.transpose())).eval();
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomposition{projected};
    if (decomposition.info() != Eigen::Success) return false;
    const Eigen::MatrixXd rotation = decomposition.eigenvectors().leftCols(count);
    vectors = space * rotation;
    mass_vectors = mass_space * rotation;
    shifted_vectors = shifted_space * rotation;
    values = decomposition.eigenvalues().head(count);
    seconds += SecondsSince(start) - (actions.Seconds - action_start);
    return true;
}

// Returns Rayleigh-Ritz pairs from a subspace and its precomputed mass and shifted actions.
bool RitzFromActions(
    Eigen::MatrixXd &space, Eigen::MatrixXd &mass_space, Eigen::MatrixXd &shifted_space,
    uint32_t count, Eigen::MatrixXd &vectors, Eigen::MatrixXd &mass_vectors,
    Eigen::MatrixXd &shifted_vectors, Eigen::VectorXd &values, double &seconds,
    const Eigen::MatrixXd &locked_vectors, const Eigen::MatrixXd &locked_mass_vectors,
    const Eigen::VectorXd &locked_values, uint32_t locked_count
) {
    const auto start = Clock::now();
    if (locked_count) {
        const Eigen::MatrixXd overlap = locked_vectors.leftCols(locked_count).transpose() * mass_space;
        space.noalias() -= locked_vectors.leftCols(locked_count) * overlap;
        mass_space.noalias() -= locked_mass_vectors.leftCols(locked_count) * overlap;
        shifted_space.noalias() -= locked_mass_vectors.leftCols(locked_count) *
            (locked_values.head(locked_count).asDiagonal() * overlap);
    }
    const Eigen::VectorXd inverse_norm =
        (space.cwiseProduct(mass_space).colwise().sum().array().sqrt().inverse()).matrix();
    if (!inverse_norm.allFinite()) return false;
    Eigen::MatrixXd gram = SymmetricCrossGram(space, mass_space);
    gram = inverse_norm.asDiagonal() * gram * inverse_norm.asDiagonal();
    gram = (0.5 * (gram + gram.transpose())).eval();
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> mass_decomposition{gram};
    if (mass_decomposition.info() != Eigen::Success) return false;
    const Eigen::VectorXd &mass_values = mass_decomposition.eigenvalues();
    const double threshold = mass_values.cwiseAbs().maxCoeff() * 1e-12;
    Eigen::Index first{};
    while (first < mass_values.size() && mass_values[first] <= threshold) ++first;
    const Eigen::Index retained = mass_values.size() - first;
    if (retained < count) return false;
    const Eigen::MatrixXd transform = inverse_norm.asDiagonal() * mass_decomposition.eigenvectors().rightCols(retained) *
        mass_values.tail(retained).cwiseSqrt().cwiseInverse().asDiagonal();
    Eigen::MatrixXd projected = SymmetricCrossGram(space, shifted_space);
    projected = transform.transpose() * projected * transform;
    projected = (0.5 * (projected + projected.transpose())).eval();
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomposition{projected};
    if (decomposition.info() != Eigen::Success) return false;
    // Combining mass orthonormalization with Ritz rotation applies one transform to each tall matrix.
    const Eigen::MatrixXd rotation = transform * decomposition.eigenvectors().leftCols(count);
    RotateTallTriple(space, mass_space, shifted_space, rotation, vectors, mass_vectors, shifted_vectors);
    values = decomposition.eigenvalues().head(count);
    seconds += SecondsSince(start);
    return true;
}

// Returns a Gaussian basis whose leading columns contain the prolonged P1 eigenbasis and four guard vectors.
Eigen::MatrixXd InitialSpace(
    const modal::FiniteCellOperator &fem, uint32_t width, double alpha,
    const modal::FiniteCellOperator::AssembledLower &p1
) {
    Eigen::MatrixXd result(fem.Dofs(), width);
    std::mt19937_64 random{20260828};
    std::normal_distribution<double> gaussian;
    for (Eigen::Index column = 0; column < result.cols(); ++column)
        for (Eigen::Index row = 0; row < result.rows(); ++row) result(row, column) = gaussian(random);
    const uint32_t count = std::min<uint32_t>(width, 3 * fem.NumP1Nodes - 1);
    if (count == 0 || count >= p1.Mass.rows()) return result;

    const uint32_t basis_size = std::min<uint32_t>(p1.Mass.rows(), std::max<uint32_t>(count + 20, 20));
    double factor_seconds{}, solve_seconds{};
    CholeskyShiftInvert operation{p1.Stiffness, p1.Mass, factor_seconds, solve_seconds};
    Spectra::SparseSymMatProd<double> mass{p1.Mass};
    Spectra::SymGEigsShiftSolver<CholeskyShiftInvert, Spectra::SparseSymMatProd<double>, Spectra::GEigsMode::ShiftInvert> eigensolver{
        operation, mass, int(count), int(basis_size), -alpha
    };
    eigensolver.init();
    eigensolver.compute(Spectra::SortRule::LargestMagn, 100, 1e-4, Spectra::SortRule::SmallestAlge);
    if (eigensolver.info() != Spectra::CompInfo::Successful) return result;
    fem.ProlongP1(eigensolver.eigenvectors().data(), result.data(), count);
    return result;
}

struct MetalPatchData {
    modal::FiniteCellMetal::SharedFloats Inverses;
    std::vector<double> Elements;
    modal::FiniteCellOperator::PackedCutOperators CutActions;
};

MetalPatchData BuildMetalPatchData(const modal::FiniteCellOperator &fem, double alpha) {
    MetalPatchData result;
    std::vector<double> elements(fem.Cells.size() * PackedLocalValues);
    struct ElementContext {
        const modal::FiniteCellOperator &Fem;
        std::vector<double> &Elements;
        double Alpha;
    } element_context{fem, elements, alpha};
    dispatch_apply_f(
        fem.Cells.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &element_context,
        [](void *raw, size_t cell) {
            auto &context = *static_cast<ElementContext *>(raw);
            context.Fem.PackCellShiftedLower(uint32_t(cell), context.Alpha, {context.Elements.data() + cell * PackedLocalValues, PackedLocalValues});
        }
    );
    result.CutActions = fem.BuildPackedCutOperators(alpha, elements);
    result.Inverses = modal::FiniteCellMetal::SharedFloats{fem.Cells.size() * PackedLocalValues};
    std::vector<uint32_t> destination(fem.Cells.size());
    uint32_t next{};
    for (uint8_t color = 0; color < 8; ++color)
        for (uint32_t cell = 0; cell < fem.Cells.size(); ++cell)
            if (fem.Cells[cell].Color == color) destination[cell] = next++;

    struct PatchContext {
        const modal::FiniteCellOperator &Fem;
        const std::vector<double> &Elements;
        MetalPatchData &Result;
        std::span<const uint32_t> Destination;
        std::atomic<bool> Failed{}, InvalidNeighborhood{};
    } context{fem, elements, result, destination};
    dispatch_apply_f(
        fem.Cells.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &context,
        [](void *raw, size_t cell) {
            auto &context = *static_cast<PatchContext *>(raw);
            const auto &target = context.Fem.Cells[cell];
            std::array<uint32_t, modal::FiniteCellOperator::NodesPerCell> neighbors;
            uint32_t neighbor_count{};
            for (uint32_t local = 0; local < modal::FiniteCellOperator::NodesPerCell; ++local) {
                const uint32_t node = target.Nodes[local];
                for (uint32_t entry = context.Fem.NodeOccurrenceOffsets[node];
                     entry < context.Fem.NodeOccurrenceOffsets[node + 1]; ++entry) {
                    const uint32_t neighbor = context.Fem.NodeOccurrences[entry] >> 5;
                    if (std::find(neighbors.begin(), neighbors.begin() + neighbor_count, neighbor) !=
                        neighbors.begin() + neighbor_count) continue;
                    if (neighbor_count == neighbors.size()) {
                        context.InvalidNeighborhood = true;
                        return;
                    }
                    neighbors[neighbor_count++] = neighbor;
                }
            }
            Eigen::MatrixXd principal = Eigen::MatrixXd::Zero(LocalDofs, LocalDofs);
            for (const uint32_t neighbor : std::span{neighbors}.first(neighbor_count)) {
                const auto &source = context.Fem.Cells[neighbor];
                std::array<int32_t, modal::FiniteCellOperator::NodesPerCell> target_local;
                target_local.fill(-1);
                for (uint32_t source_local = 0; source_local < modal::FiniteCellOperator::NodesPerCell; ++source_local) {
                    const auto first = target.Nodes.begin(), last = target.Nodes.end();
                    const auto found = std::find(first, last, source.Nodes[source_local]);
                    if (found != last) target_local[source_local] = int32_t(found - first);
                }
                const size_t element_offset = size_t(neighbor) * PackedLocalValues;
                for (uint32_t a = 0; a < modal::FiniteCellOperator::NodesPerCell; ++a) {
                    if (target_local[a] < 0) continue;
                    for (uint32_t c = 0; c < modal::FiniteCellOperator::NodesPerCell; ++c) {
                        if (target_local[c] < 0) continue;
                        for (uint32_t p = 0; p < 3; ++p)
                            for (uint32_t q = 0; q < 3; ++q) {
                                const uint32_t row = 3 * a + p, column = 3 * c + q;
                                const uint32_t lower_row = std::max(row, column);
                                const uint32_t lower_column = std::min(row, column);
                                principal(3 * target_local[a] + p, 3 * target_local[c] + q) +=
                                    context.Elements[element_offset + size_t(lower_row) * (lower_row + 1) / 2 + lower_column];
                            }
                    }
                }
            }
            const Eigen::LLT<Eigen::MatrixXd> factor{principal};
            if (factor.info() != Eigen::Success) {
                context.Failed = true;
                return;
            }
            const Eigen::MatrixXd inverse = factor.solve(Eigen::MatrixXd::Identity(LocalDofs, LocalDofs));
            const size_t inverse_offset = size_t(context.Destination[cell]) * PackedLocalValues;
            for (uint32_t row = 0; row < LocalDofs; ++row)
                for (uint32_t column = 0; column <= row; ++column)
                    context.Result.Inverses.Values()[inverse_offset + size_t(row) * (row + 1) / 2 + column] = float(inverse(row, column));
        }
    );
    if (context.InvalidNeighborhood)
        throw std::runtime_error("Finite-cell patch neighborhood exceeds the local storage bound.");
    if (context.Failed)
        throw std::runtime_error("Finite-cell Metal patch block is not positive definite.");
    result.Elements = std::move(elements);
    return result;
}

// Applies forward and reverse FP32 patch sweeps around a resident P1 multigrid correction and converts the result to FP64.
struct MetalMultiplicativePreconditioner {
    std::unique_ptr<modal::FiniteCellMetal> Metal;
    modal::FiniteCellOperator::PackedCutOperators CutActions;
    mutable modal::FiniteCellMetal::Block Fine, Remaining, Result, CoarseResidual, CoarseCorrection, Scratch;
    mutable uint32_t Width{};

    MetalMultiplicativePreconditioner(
        const modal::FiniteCellOperator &fem, double alpha,
        const modal::FiniteCellOperator::AssembledLower &p1_assembly
    ) {
        auto multigrid_future = std::async(std::launch::async, [&] {
            return modal::FiniteCellMetal::PrepareP1Multigrid(fem, alpha, p1_assembly);
        });
        auto metal_future = std::async(std::launch::async, [&] { return std::make_unique<modal::FiniteCellMetal>(fem); });
        auto patch = BuildMetalPatchData(fem, alpha);
        Metal = metal_future.get();
        CutActions = std::move(patch.CutActions);
        Metal->ConfigurePackedPatch(std::move(patch.Inverses), patch.Elements);
        Metal->ConfigureP1Multigrid(multigrid_future.get());
    }

    void Apply(const double *input, double *output, uint32_t width) const {
        if (!width) return;
        if (Width != width) {
            Fine = Metal->CreateSharedBlock(width);
            Remaining = Metal->CreateBlock(width);
            Result = Metal->CreateBlock(width);
            CoarseResidual = Metal->CreateP1Block(width);
            CoarseCorrection = Metal->CreateP1Block(width);
            Scratch = Metal->CreateBlock(width);
            Width = width;
        }
        Metal->Upload(Fine, input);
        Metal->ApplyPackedLocalizedMultiplicativePatchSweep(Fine, Result, Remaining);
        Metal->RestrictP1(Remaining, CoarseResidual);
        Metal->ApplyP1Multigrid(CoarseResidual, CoarseCorrection);
        Metal->ProlongP1(CoarseCorrection, Fine);
        Metal->LinearCombination(Result, 1, Fine, 1, Result);
        Metal->ApplyElement(Fine, Scratch);
        Metal->LinearCombination(Remaining, 1, Scratch, -1, Remaining);
        Metal->ApplyPackedLocalizedMultiplicativePatchSweep(Remaining, Scratch, Fine, true, false);
        Metal->LinearCombination(Result, 1, Scratch, 1, Result);
        Metal->LinearCombination(Result, 1.1f, Fine, 0, Fine);
        Metal->Download(Fine, output);
    }
};

} // namespace

namespace {
modal::FiniteCellBlockResult SolvePreferred(
    const modal::FiniteCellOperator &fem, uint32_t count, double alpha, double tolerance,
    uint32_t max_iterations
) {
    max_iterations = std::min(100u, max_iterations);
    constexpr uint32_t guard_vectors{4}, stagnation_window{12};
    constexpr double stagnation_reduction{0.8};
    const auto start = Clock::now();
    modal::FiniteCellBlockResult result;
    Actions actions{fem};
    const auto apply_mass_shifted = [&](const double *input, double *mass, double *shifted, uint32_t width) {
        actions.ApplyMassShifted(input, mass, shifted, width, alpha);
    };
    double preconditioner_setup_seconds{}, preconditioner_seconds{}, ritz_seconds{};
    double initialization_seconds{}, residual_seconds{}, recurrence_seconds{}, certification_seconds{};
    const auto finish = [&] {
        result.Profile.Total = SecondsSince(start);
        result.Profile.Actions = actions.Seconds;
        result.Profile.ActionSetup = actions.PackedCut ? actions.PackedCut->BuildSeconds : 0;
        result.Profile.Initialization = initialization_seconds;
        result.Profile.PreconditionerSetup = preconditioner_setup_seconds;
        result.Profile.Preconditioner = preconditioner_seconds;
        result.Profile.RayleighRitz = ritz_seconds;
        result.Profile.Residuals = residual_seconds;
        result.Profile.Recurrence = recurrence_seconds;
        result.Profile.Certification = certification_seconds;
        result.Profile.Other = result.Profile.Total - result.Profile.Actions - result.Profile.PreconditionerSetup -
            result.Profile.Preconditioner - result.Profile.RayleighRitz;
        return result;
    };
    if (count == 0 || count > fem.Dofs()) return finish();

    const auto initialization_start = Clock::now();
    // The initial space and multigrid hierarchy share one temporary P1 assembly.
    std::optional p1_assembly{fem.AssembleP1Lower()};
    auto preconditioner_setup_future = std::async(std::launch::async, [&] {
        return MetalMultiplicativePreconditioner{fem, alpha, *p1_assembly};
    });
    const uint32_t requested_count = count;
    uint32_t width = std::min<uint32_t>(fem.Dofs(), count + std::min<uint32_t>(guard_vectors, count));
    Eigen::MatrixXd space = InitialSpace(fem, width, alpha, *p1_assembly);
    initialization_seconds += SecondsSince(initialization_start);
    Eigen::MatrixXd mass_space, vectors, mass_vectors, shifted_vectors;
    Eigen::VectorXd shifted_values;
    if (!Ritz(actions, alpha, space, mass_space, width, vectors, mass_vectors, shifted_vectors, shifted_values, ritz_seconds)) {
        result.Iterations = std::numeric_limits<uint32_t>::max();
        return finish();
    }
    uint32_t locked_count = count > 6 ? 6 : 0;
    Eigen::MatrixXd locked_vectors, locked_mass_vectors;
    Eigen::VectorXd locked_values;
    if (locked_count) {
        locked_vectors.resize(fem.Dofs(), requested_count);
        locked_mass_vectors.resize(fem.Dofs(), requested_count);
        locked_values.resize(requested_count);
        locked_vectors.leftCols(locked_count) = vectors.leftCols(locked_count);
        locked_mass_vectors.leftCols(locked_count) = mass_vectors.leftCols(locked_count);
        locked_values.head(locked_count) = shifted_values.head(locked_count);
        RemoveLeadingColumns(vectors, locked_count);
        RemoveLeadingColumns(mass_vectors, locked_count);
        RemoveLeadingColumns(shifted_vectors, locked_count);
        RemoveLeadingEntries(shifted_values, locked_count);
        count -= locked_count;
        width -= locked_count;
    }
    const auto relative_residual = [alpha](const auto &mass, const auto &shifted, const auto &residual, double shifted_value) {
        const double scale = (shifted - alpha * mass).norm() + std::abs(shifted_value - alpha) * mass.norm();
        return scale == 0 ? residual.norm() : residual.norm() / scale;
    };
    // The six rigid-body eigenvalues equal the shift within the configured relative threshold.
    const auto rigid = [&](uint32_t mode) { return std::abs(shifted_values[mode] - alpha) <= alpha * 1e-4; };
    const auto accept = [&](const Eigen::VectorXd &active_relative) {
        result.Eigenvalues.resize(requested_count);
        result.Eigenvectors.resize(fem.Dofs(), requested_count);
        result.RelativeResiduals.resize(requested_count);
        if (locked_count) {
            result.Eigenvalues.head(locked_count) = locked_values.array() - alpha;
            result.Eigenvectors.leftCols(locked_count) = locked_vectors.leftCols(locked_count);
        }
        result.Eigenvalues.tail(count) = shifted_values.head(count).array() - alpha;
        result.Eigenvectors.rightCols(count) = vectors.leftCols(count);
        result.RelativeResiduals.tail(count) = active_relative.head(count);
        const auto certification_start = Clock::now();
        result.Certification = CertifyFiniteCellEigenpairs(fem, result.Eigenvalues, result.Eigenvectors);
        if (locked_count)
            result.RelativeResiduals.head(locked_count) = result.Certification.RelativeResiduals.head(locked_count);
        certification_seconds += SecondsSince(certification_start);
    };
    const auto wait_start = Clock::now();
    auto preconditioner = preconditioner_setup_future.get();
    preconditioner_setup_seconds = SecondsSince(wait_start);
    p1_assembly.reset();
    actions.PackedCut = std::move(preconditioner.CutActions);
    Eigen::MatrixXf compact_previous_direction;
    std::vector<double> stagnation_history;

    for (uint32_t iteration = 0; iteration < max_iterations; ++iteration) {
        const auto residual_start = Clock::now();
        const double residual_action_start = actions.Seconds;
        Eigen::MatrixXd residual = shifted_vectors - mass_vectors * shifted_values.asDiagonal();
        Eigen::VectorXd relative(width);
        const auto update_relative = [&] {
            for (uint32_t i = 0; i < width; ++i)
                relative[i] = relative_residual(
                    mass_vectors.col(i), shifted_vectors.col(i), residual.col(i), shifted_values[i]
                );
        };
        update_relative();
        result.Iterations = iteration;

        const auto is_converged = [&] {
            for (uint32_t i = 0; i < count; ++i)
                if (!rigid(i) && relative[i] >= tolerance) return false;
            return true;
        };
        bool converged = is_converged();
        if (converged) {
            // Recompute the residual from exact actions before accepting a recurrence-driven convergence.
            actions.ApplyMassShifted(vectors.data(), mass_vectors.data(), shifted_vectors.data(), width, alpha);
            residual = shifted_vectors - mass_vectors * shifted_values.asDiagonal();
            update_relative();
            converged = is_converged();
        }
        residual_seconds += SecondsSince(residual_start) - (actions.Seconds - residual_action_start);
        if (converged) {
            accept(relative);
            return finish();
        }

        uint32_t newly_locked{};
        while (newly_locked < count && relative[newly_locked] < tolerance) ++newly_locked;
        if (newly_locked) stagnation_history.clear();
        else {
            double maximum{};
            for (uint32_t i = 0; i < count; ++i)
                if (!rigid(i)) maximum = std::max(maximum, relative[i]);
            stagnation_history.push_back(maximum);
            if (stagnation_history.size() > stagnation_window + 1) stagnation_history.erase(stagnation_history.begin());
            if (stagnation_history.size() == stagnation_window + 1) {
                const double best = *std::ranges::min_element(stagnation_history);
                const bool stalled = best >= stagnation_reduction * stagnation_history.front();
                const double log_reduction = std::log(best / stagnation_history.front());
                const double projected_iterations = best > tolerance && log_reduction < 0 ?
                    stagnation_window * std::log(tolerance / best) / log_reduction :
                    0;
                const bool cannot_finish = best > 4 * tolerance && projected_iterations > max_iterations - iteration;
                if (stalled || cannot_finish) {
                    result.Profile.Stagnated = true;
                    result.Profile.StagnationIteration = iteration;
                    result.Profile.StagnationResidual = maximum;
                    accept(relative);
                    return finish();
                }
            }
        }

        const auto recurrence_start = Clock::now();
        const double recurrence_action_start = actions.Seconds;
        const double recurrence_preconditioner_start = preconditioner_seconds;
        const double recurrence_ritz_start = ritz_seconds;

        if (newly_locked) {
            const uint32_t old_locked = locked_count;
            locked_count += newly_locked;
            locked_vectors.middleCols(old_locked, newly_locked) = vectors.leftCols(newly_locked);
            locked_mass_vectors.middleCols(old_locked, newly_locked) = mass_vectors.leftCols(newly_locked);
            locked_values.segment(old_locked, newly_locked) = shifted_values.head(newly_locked);
            RemoveLeadingColumns(vectors, newly_locked);
            RemoveLeadingColumns(mass_vectors, newly_locked);
            RemoveLeadingColumns(shifted_vectors, newly_locked);
            RemoveLeadingEntries(shifted_values, newly_locked);
            RemoveLeadingColumns(residual, newly_locked);
            RemoveLeadingEntries(relative, newly_locked);
            count -= newly_locked;
            width -= newly_locked;
            compact_previous_direction.resize(0, 0);
        }

        std::vector<uint32_t> active;
        active.reserve(width);
        for (uint32_t i = 0; i < width; ++i)
            if (i >= count || (!rigid(i) && relative[i] >= tolerance)) active.push_back(i);
        Eigen::MatrixXd active_residual(fem.Dofs(), active.size());
        for (uint32_t i = 0; i < active.size(); ++i) active_residual.col(i) = residual.col(active[i]);
        residual.resize(0, 0);

        const Eigen::Index correction_columns = Eigen::Index(active.size());
        const Eigen::Index history_columns = compact_previous_direction.cols();
        space.resize(fem.Dofs(), width + correction_columns + history_columns);
        mass_space.resizeLike(space);
        Eigen::MatrixXd shifted_space(fem.Dofs(), space.cols());
        space.leftCols(width) = vectors;
        mass_space.leftCols(width) = mass_vectors;
        shifted_space.leftCols(width) = shifted_vectors;
        const auto preconditioner_start = Clock::now();
        const double action_start = actions.Seconds;
        preconditioner.Apply(
            active_residual.data(), space.middleCols(width, correction_columns).data(), uint32_t(active.size())
        );
        preconditioner_seconds += SecondsSince(preconditioner_start) - (actions.Seconds - action_start);
        active_residual.resize(0, 0);
        apply_mass_shifted(
            space.middleCols(width, correction_columns).data(),
            mass_space.middleCols(width, correction_columns).data(),
            shifted_space.middleCols(width, correction_columns).data(), uint32_t(correction_columns)
        );
        if (history_columns) {
            space.rightCols(history_columns) = compact_previous_direction.cast<double>();
            apply_mass_shifted(
                space.rightCols(history_columns).data(), mass_space.rightCols(history_columns).data(),
                shifted_space.rightCols(history_columns).data(), uint32_t(history_columns)
            );
        }
        if (!RitzFromActions(
                space, mass_space, shifted_space, width, vectors, mass_vectors, shifted_vectors,
                shifted_values, ritz_seconds, locked_vectors, locked_mass_vectors, locked_values, locked_count
            )) {
            result.Iterations = iteration + 1;
            return finish();
        }
        compact_previous_direction.resize(fem.Dofs(), active.size());
        for (uint32_t i = 0; i < active.size(); ++i) {
            const auto vector = vectors.col(active[i]);
            compact_previous_direction.col(i) = (vector - space.leftCols(width) * (mass_space.leftCols(width).transpose() * vector)).cast<float>();
        }

        recurrence_seconds += SecondsSince(recurrence_start) - (actions.Seconds - recurrence_action_start) -
            (preconditioner_seconds - recurrence_preconditioner_start) - (ritz_seconds - recurrence_ritz_start);
        result.Iterations = iteration + 1;
    }

    const auto residual_start = Clock::now();
    const Eigen::MatrixXd residual = shifted_vectors.leftCols(count) - mass_vectors.leftCols(count) * shifted_values.head(count).asDiagonal();
    Eigen::VectorXd relative(count);
    for (uint32_t i = 0; i < count; ++i)
        relative[i] = relative_residual(
            mass_vectors.col(i), shifted_vectors.col(i), residual.col(i), shifted_values[i]
        );
    residual_seconds += SecondsSince(residual_start);
    accept(relative);
    return finish();
}

} // namespace

modal::FiniteCellBlockResult modal::SolveFiniteCellBlock(
    const FiniteCellOperator &fem, uint32_t count, double alpha, double tolerance, uint32_t max_iterations
) {
    const auto start = Clock::now();
    auto preferred = SolvePreferred(fem, count, alpha, tolerance, max_iterations);
    const uint32_t first_physical = std::min(6u, count), physical_count = count - first_physical;
    const auto certified = [&](const FiniteCellBlockResult &result) {
        return result.Eigenvalues.size() == count && result.Certification.RelativeResiduals.size() == count &&
            result.Eigenvalues.allFinite() && result.Certification.RelativeResiduals.allFinite() &&
            result.Certification.MassOrthogonalityError < 1e-9 &&
            (!physical_count || result.Certification.RelativeResiduals.segment(first_physical, physical_count).maxCoeff() < tolerance);
    };
    if (certified(preferred)) return preferred;
    const double attempt_seconds = SecondsSince(start);
    const uint32_t attempt_iterations = preferred.Iterations;
    const bool attempt_stagnated = preferred.Profile.Stagnated;
    const bool attempt_certified = preferred.Certification.RelativeResiduals.size() == count;
    const double attempt_residual = attempt_certified && physical_count ?
        preferred.Certification.RelativeResiduals.segment(first_physical, physical_count).maxCoeff() :
        physical_count ? std::numeric_limits<double>::infinity() :
                         0;
    const double attempt_orthogonality = attempt_certified ? preferred.Certification.MassOrthogonalityError :
                                                             std::numeric_limits<double>::infinity();
    preferred = {};
    auto cholesky = oracle::SolveCholesky(fem, count, alpha, tolerance, max_iterations);
    if (!certified(cholesky)) throw std::runtime_error("Finite-cell default and fallback solvers did not certify within the iteration budget.");
    cholesky.Profile.FallbackAttempt = attempt_seconds;
    cholesky.Profile.FallbackAttemptIterations = attempt_iterations;
    cholesky.Profile.FallbackAttemptStagnated = attempt_stagnated;
    cholesky.Profile.FallbackAttemptResidual = attempt_residual;
    cholesky.Profile.FallbackAttemptOrthogonality = attempt_orthogonality;
    cholesky.Profile.Total += attempt_seconds;
    return cholesky;
}

modal::FiniteCellBlockResult modal::oracle::SolveCholesky(
    const FiniteCellOperator &fem, uint32_t count, double alpha, double tolerance,
    uint32_t max_iterations
) {
    const auto start = Clock::now();
    FiniteCellBlockResult result;
    if (!count || count >= fem.Dofs()) return result;
    const auto assembly_start = Clock::now();
    const auto assembled = fem.AssembleLower();
    result.Profile.Actions = SecondsSince(assembly_start);
    double factor_seconds{}, solve_seconds{};
    CholeskyShiftInvert inverse{assembled.Stiffness, assembled.Mass, factor_seconds, solve_seconds};
    Spectra::SparseSymMatProd<double> mass{assembled.Mass};
    const uint32_t solve_count = std::min<uint32_t>(fem.Dofs() - 1, count + 4);
    const uint32_t basis = std::min<uint32_t>(
        fem.Dofs(), std::max(2 * solve_count + 20, solve_count + 40)
    );
    Spectra::SymGEigsShiftSolver<CholeskyShiftInvert, Spectra::SparseSymMatProd<double>, Spectra::GEigsMode::ShiftInvert> eigensolver{
        inverse, mass, int(solve_count), int(basis), -alpha
    };
    eigensolver.init();
    eigensolver.compute(
        Spectra::SortRule::LargestMagn, int(max_iterations), 0.01 * tolerance,
        Spectra::SortRule::SmallestAlge
    );
    result.Iterations = uint32_t(eigensolver.num_iterations());
    if (eigensolver.info() == Spectra::CompInfo::Successful) {
        Eigen::MatrixXd space = eigensolver.eigenvectors();
        result.Eigenvalues = eigensolver.eigenvalues().head(count);
        result.Eigenvectors = space.leftCols(count);
        result.Certification = CertifyFiniteCellEigenpairs(fem, result.Eigenvalues, result.Eigenvectors);
        result.RelativeResiduals = result.Certification.RelativeResiduals;
        const uint32_t first_physical = std::min(6u, count), physical_count = count - first_physical;
        const bool safely_certified = result.Certification.MassOrthogonalityError < 0.5e-9 &&
            (!physical_count || result.RelativeResiduals.segment(first_physical, physical_count).maxCoeff() < 0.5 * tolerance);
        if (!safely_certified) {
            Eigen::MatrixXd mass_space, vectors, mass_vectors, shifted_vectors;
            Eigen::VectorXd shifted_values;
            Actions actions{fem};
            if (Ritz(
                    actions, alpha, space, mass_space, solve_count, vectors, mass_vectors,
                    shifted_vectors, shifted_values, result.Profile.RayleighRitz
                )) {
                const Eigen::MatrixXd stiffness_vectors = shifted_vectors - alpha * mass_vectors;
                const Eigen::MatrixXd residual = shifted_vectors - mass_vectors * shifted_values.asDiagonal();
                std::vector<uint32_t> active;
                for (uint32_t mode = first_physical; mode < count; ++mode) {
                    const double scale = stiffness_vectors.col(mode).norm() +
                        std::abs(shifted_values[mode] - alpha) * mass_vectors.col(mode).norm();
                    const double relative = scale == 0 ? residual.col(mode).norm() : residual.col(mode).norm() / scale;
                    if (relative >= 0.5 * tolerance) active.push_back(mode);
                }
                Eigen::MatrixXd refined_vectors;
                Eigen::VectorXd refined_values;
                bool refined = active.empty();
                if (refined) {
                    refined_vectors = vectors.leftCols(count);
                    refined_values = shifted_values.head(count).array() - alpha;
                } else {
                    Eigen::MatrixXd active_residual(fem.Dofs(), active.size());
                    for (uint32_t column = 0; column < active.size(); ++column)
                        active_residual.col(column) = residual.col(active[column]);
                    Eigen::MatrixXd correction(fem.Dofs(), active.size());
                    inverse.solve_panel(active_residual.data(), correction.data(), int(active.size()));
                    space.resize(fem.Dofs(), solve_count + active.size());
                    space.leftCols(solve_count) = vectors;
                    space.rightCols(active.size()) = correction;
                    refined = Ritz(
                        actions, alpha, space, mass_space, count, refined_vectors, mass_vectors,
                        shifted_vectors, shifted_values, result.Profile.RayleighRitz
                    );
                    if (refined) refined_values = shifted_values.array() - alpha;
                }
                if (refined) {
                    result.Eigenvalues = std::move(refined_values);
                    result.Eigenvectors = std::move(refined_vectors);
                    result.Certification = CertifyFiniteCellEigenpairs(fem, result.Eigenvalues, result.Eigenvectors);
                    result.RelativeResiduals = result.Certification.RelativeResiduals;
                }
                result.Profile.Actions += actions.Seconds;
            }
        }
    }
    result.Profile.PreconditionerSetup = factor_seconds;
    result.Profile.Preconditioner = solve_seconds;
    result.Profile.Total = SecondsSince(start);
    result.Profile.Other = result.Profile.Total - result.Profile.Actions - factor_seconds - solve_seconds - result.Profile.RayleighRitz;
    return result;
}
