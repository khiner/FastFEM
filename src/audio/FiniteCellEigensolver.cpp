#include "FiniteCellEigensolver.h"

#include "GeneralizedEigenSolver.h"
#include "finite_cell/AccelerateShiftInvert.h"
#include "finite_cell/AssembledEigensolver.h"
#include "finite_cell/MetalOperations.h"
#include "numeric/Accelerate.h"

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

void RemoveLeadingColumns(numeric::Matrix<double> &matrix, uint32_t count) {
    const size_t remaining = matrix.cols() - count;
    std::memmove(matrix.data(), matrix.data() + matrix.rows() * count, matrix.rows() * remaining * sizeof(double));
    matrix.Resize(matrix.rows(), remaining);
}

void RemoveLeadingEntries(numeric::Vector<double> &vector, uint32_t count) {
    const size_t remaining = vector.size() - count;
    std::memmove(vector.data(), vector.data() + count, remaining * sizeof(double));
    vector.Resize(remaining);
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
        Seconds += ApplyMassShiftedTimed(input, mass_output, shifted_output, width, alpha);
    }

    double ApplyMassShiftedTimed(
        const double *input, double *mass_output, double *shifted_output, uint32_t width, double alpha
    ) {
        const auto start = Clock::now();
        if (PackedCut) Fem.ApplyMassShiftedExpandedPackedCut(*PackedCut, input, mass_output, shifted_output, width);
        else Fem.ApplyMassShifted(input, mass_output, shifted_output, width, alpha);
        return SecondsSince(start);
    }

    void Measure(auto &&operation) {
        const auto start = Clock::now();
        operation();
        Seconds += SecondsSince(start);
    }
};

numeric::Matrix<double> SymmetricCrossGram(const numeric::Matrix<double> &a, const numeric::Matrix<double> &b) {
    if (a.rows() != b.rows() || a.cols() != b.cols())
        throw std::invalid_argument("Symmetric cross-Gram operands must have equal dimensions.");
    numeric::Matrix<double> result(a.cols(), a.cols());
    numeric::SymmetricCrossGram(a.data(), b.data(), result.data(), uint32_t(a.rows()), uint32_t(a.cols()));
    return result;
}

void SetResiduals(modal::FiniteCellEigenpairs &result, const numeric::Matrix<double> &mass_vectors, const numeric::Matrix<double> &shifted_vectors, double alpha) {
    const size_t count = result.Eigenvalues.size();
    const auto mass = mass_vectors.FirstColumns(count);
    numeric::Matrix<double> stiffness = numeric::Copy(shifted_vectors.FirstColumns(count));
    numeric::AddScaled(-alpha, mass, stiffness.View());
    const numeric::Matrix<double> residual = numeric::ColumnScaledDifference(stiffness.View(), mass, result.Eigenvalues.View());
    result.RelativeResiduals.Resize(count);
    for (size_t mode = 0; mode < count; ++mode) {
        const double scale = numeric::Norm(stiffness.Column(mode)) + std::abs(result.Eigenvalues[mode]) * numeric::Norm(mass.Column(mode));
        const double residual_norm = numeric::Norm(residual.Column(mode));
        result.RelativeResiduals[mode] = scale == 0 ? residual_norm : residual_norm / scale;
    }
}

void RotateTallTriple(
    const numeric::Matrix<double> &a, const numeric::Matrix<double> &b, const numeric::Matrix<double> &c,
    const numeric::Matrix<double> &rotation, numeric::Matrix<double> &a_rotated, numeric::Matrix<double> &b_rotated,
    numeric::Matrix<double> &c_rotated
) {
    a_rotated.Resize(a.rows(), rotation.cols());
    b_rotated.Resize(b.rows(), rotation.cols());
    c_rotated.Resize(c.rows(), rotation.cols());
    auto b_future = std::async(std::launch::async, [&] { numeric::Multiply(b.View(), rotation.View(), b_rotated.View()); });
    auto c_future = std::async(std::launch::async, [&] { numeric::Multiply(c.View(), rotation.View(), c_rotated.View()); });
    numeric::Multiply(a.View(), rotation.View(), a_rotated.View());
    b_future.get();
    c_future.get();
}

// Replaces `vectors` with an M-orthonormal basis and returns false when fewer than `required` independent directions remain.
bool MOrthonormalize(Actions &actions, numeric::Matrix<double> &vectors, numeric::Matrix<double> &mass_vectors, uint32_t required) {
    mass_vectors.Resize(actions.Fem.Dofs(), vectors.cols());
    actions.ApplyMass(vectors.data(), mass_vectors.data(), uint32_t(vectors.cols()));
    numeric::Vector<double> inverse_norm(vectors.cols());
    for (size_t column = 0; column < vectors.cols(); ++column)
        inverse_norm[column] = 1 / std::sqrt(numeric::Dot(vectors.Column(column), mass_vectors.Column(column)));
    if (!numeric::AllFinite(inverse_norm.View())) return false;
    auto mass_scale = std::async(std::launch::async, [&] { numeric::ScaleColumns(mass_vectors.View(), inverse_norm.View()); });
    numeric::ScaleColumns(vectors.View(), inverse_norm.View());
    mass_scale.get();
    numeric::Matrix<double> gram = numeric::TransposeMultiply(vectors.View(), mass_vectors.View());
    numeric::Symmetrize(gram.View());
    numeric::Vector<double> values;
    if (!modal::eigensolver::SelfAdjointEigenvectors(gram, values)) return false;
    const double threshold = numeric::MaximumAbsolute(values.View()) * 1e-12;
    size_t first{};
    while (first < values.size() && values[first] <= threshold) ++first;
    const size_t retained = values.size() - first;
    if (retained < required) return false;
    numeric::Matrix<double> transform = numeric::Copy(gram.LastColumns(retained));
    numeric::Vector<double> inverse_sqrt(retained);
    for (size_t column = 0; column < retained; ++column) inverse_sqrt[column] = 1 / std::sqrt(values[first + column]);
    numeric::ScaleColumns(transform.View(), inverse_sqrt.View());
    numeric::Matrix<double> rotated_mass;
    auto mass_rotation = std::async(std::launch::async, [&] { rotated_mass = numeric::Multiply(mass_vectors.View(), transform.View()); });
    vectors = numeric::Multiply(vectors.View(), transform.View());
    mass_rotation.get();
    mass_vectors = std::move(rotated_mass);
    return true;
}

bool Ritz(
    Actions &actions, double alpha, numeric::Matrix<double> &space, numeric::Matrix<double> &mass_space, uint32_t count,
    numeric::Matrix<double> &vectors, numeric::Matrix<double> &mass_vectors, numeric::Matrix<double> &shifted_vectors,
    numeric::Vector<double> &values, double &seconds
) {
    const auto start = Clock::now();
    const double action_start = actions.Seconds;
    if (!MOrthonormalize(actions, space, mass_space, count)) return false;
    numeric::Matrix<double> shifted_space(actions.Fem.Dofs(), space.cols());
    actions.ApplyShifted(space.data(), shifted_space.data(), uint32_t(space.cols()), alpha);
    numeric::Matrix<double> projected = numeric::TransposeMultiply(space.View(), shifted_space.View());
    numeric::Symmetrize(projected.View());
    numeric::Vector<double> projected_values;
    if (!modal::eigensolver::SelfAdjointEigenvectors(projected, projected_values)) return false;
    numeric::Matrix<double> rotation = numeric::Copy(projected.FirstColumns(count));
    RotateTallTriple(space, mass_space, shifted_space, rotation, vectors, mass_vectors, shifted_vectors);
    values = numeric::Copy(projected_values.First(count));
    seconds += SecondsSince(start) - (actions.Seconds - action_start);
    return true;
}

// Returns Rayleigh-Ritz pairs from a subspace and its precomputed mass and shifted actions.
bool RitzFromActions(
    numeric::Matrix<double> &space, numeric::Matrix<double> &mass_space, numeric::Matrix<double> &shifted_space,
    uint32_t count, numeric::Matrix<double> &vectors, numeric::Matrix<double> &mass_vectors,
    numeric::Matrix<double> &shifted_vectors, numeric::Vector<double> &values, double &seconds,
    const numeric::Matrix<double> &locked_vectors, const numeric::Matrix<double> &locked_mass_vectors,
    const numeric::Vector<double> &locked_values, uint32_t locked_count
) {
    const auto start = Clock::now();
    if (locked_count) {
        const numeric::Matrix<double> overlap = numeric::TransposeMultiply(locked_vectors.FirstColumns(locked_count), mass_space.View());
        auto mass_projection = std::async(std::launch::async, [&] {
            numeric::SubtractProduct(mass_space.View(), locked_mass_vectors.FirstColumns(locked_count), overlap.View());
        });
        auto shifted_projection = std::async(std::launch::async, [&] {
            numeric::Matrix<double> weighted = numeric::Copy(overlap.View());
            modal::eigensolver::ScaleRows(weighted.View(), locked_values.First(locked_count));
            numeric::SubtractProduct(shifted_space.View(), locked_mass_vectors.FirstColumns(locked_count), weighted.View());
        });
        numeric::SubtractProduct(space.View(), locked_vectors.FirstColumns(locked_count), overlap.View());
        mass_projection.get();
        shifted_projection.get();
    }
    auto mass_gram = std::async(std::launch::async, [&] { return SymmetricCrossGram(space, mass_space); });
    numeric::Matrix<double> projected = SymmetricCrossGram(space, shifted_space);
    numeric::Matrix<double> gram = mass_gram.get();
    numeric::Vector<double> inverse_norm(space.cols());
    for (size_t column = 0; column < space.cols(); ++column)
        inverse_norm[column] = 1 / std::sqrt(numeric::Dot(space.Column(column), mass_space.Column(column)));
    if (!numeric::AllFinite(inverse_norm.View())) return false;
    numeric::ScaleRowsAndColumns(gram.View(), inverse_norm.View());
    numeric::Symmetrize(gram.View());
    numeric::Vector<double> mass_values;
    if (!modal::eigensolver::SelfAdjointEigenvectors(gram, mass_values)) return false;
    const double threshold = numeric::MaximumAbsolute(mass_values.View()) * 1e-12;
    size_t first{};
    while (first < mass_values.size() && mass_values[first] <= threshold) ++first;
    const size_t retained = mass_values.size() - first;
    if (retained < count) return false;
    numeric::Matrix<double> transform = numeric::Copy(gram.LastColumns(retained));
    modal::eigensolver::ScaleRows(transform.View(), inverse_norm.View());
    numeric::Vector<double> inverse_sqrt(retained);
    for (size_t column = 0; column < retained; ++column) inverse_sqrt[column] = 1 / std::sqrt(mass_values[first + column]);
    numeric::ScaleColumns(transform.View(), inverse_sqrt.View());
    projected = numeric::TransposeMultiply(transform.View(), numeric::Multiply(projected.View(), transform.View()).View());
    numeric::Symmetrize(projected.View());
    numeric::Vector<double> projected_values;
    if (!modal::eigensolver::SelfAdjointEigenvectors(projected, projected_values)) return false;
    // Combining mass orthonormalization with Ritz rotation applies one transform to each tall matrix.
    const numeric::Matrix<double> rotation = numeric::Multiply(transform.View(), projected.FirstColumns(count));
    RotateTallTriple(space, mass_space, shifted_space, rotation, vectors, mass_vectors, shifted_vectors);
    values = numeric::Copy(projected_values.First(count));
    seconds += SecondsSince(start);
    return true;
}

// Returns a Gaussian basis whose leading columns contain the prolonged P1 eigenbasis and four guard vectors.
numeric::Matrix<double> InitialSpace(
    const modal::FiniteCellOperator &fem, uint32_t width, double alpha,
    const modal::AssembledPencil &p1
) {
    numeric::Matrix<double> result(fem.Dofs(), width);
    std::mt19937_64 random{20260828};
    std::normal_distribution<double> gaussian;
    const auto fill_random = [&](uint32_t first) {
        for (size_t column = first; column < result.cols(); ++column)
            for (size_t row = 0; row < result.rows(); ++row) result(row, column) = gaussian(random);
    };
    const uint32_t count = std::min<uint32_t>(width, 3 * fem.NumP1Nodes - 1);
    if (count == 0 || count >= uint32_t(p1.Mass.rows())) {
        fill_random(0);
        return result;
    }

    const uint32_t basis_size = std::min<uint32_t>(p1.Mass.rows(), std::max<uint32_t>(count + 2, 20));
    const double seed_tolerance = fem.Dofs() < 20'000 ? 1e-2 : 1e-1; // Fine iterations above 20k DOFs cost more than the additional P1 accuracy.
    double factor_seconds{}, solve_seconds{};
    modal::finite_cell::AccelerateShiftInvert operation{p1.Stiffness, p1.Mass, factor_seconds, solve_seconds};
    const auto eigensolver = modal::eigensolver::SolveGeneralizedInverseIteration(
        operation, p1.Mass, p1.Stiffness,
        {
            .Count = count,
            .SubspaceSize = basis_size,
            .Shift = -alpha,
            .IterationTolerance = 1e-1,
            .ResidualTolerance = seed_tolerance,
            .MaxIterations = 100,
            .MaxRefinementIterations = 20,
            .RandomSeed = 20260828,
        }
    );
    if (!eigensolver.Converged) {
        fill_random(0);
        return result;
    }
    fem.ProlongP1(eigensolver.Eigenvectors.data(), result.data(), count);
    fill_random(count);
    return result;
}

struct MetalPatchData {
    modal::finite_cell::MetalOperations::SharedFloats Inverses;
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
    result.Inverses = modal::finite_cell::MetalOperations::SharedFloats{fem.Cells.size() * PackedLocalValues};
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
            numeric::Matrix<double> principal(LocalDofs, LocalDofs);
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
            if (!numeric::CholeskyInverse(principal.data(), LocalDofs)) {
                context.Failed = true;
                return;
            }
            const size_t inverse_offset = size_t(context.Destination[cell]) * PackedLocalValues;
            for (uint32_t row = 0; row < LocalDofs; ++row)
                for (uint32_t column = 0; column <= row; ++column)
                    context.Result.Inverses.Values()[inverse_offset + size_t(row) * (row + 1) / 2 + column] = float(principal(row, column));
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
    std::unique_ptr<modal::finite_cell::MetalOperations> Metal;
    modal::FiniteCellOperator::PackedCutOperators CutActions;
    mutable modal::finite_cell::MetalOperations::Block Fine, Remaining, Result, CoarseResidual, CoarseCorrection, Scratch;
    mutable uint32_t Width{};

    MetalMultiplicativePreconditioner(
        const modal::FiniteCellOperator &fem, double alpha,
        const modal::AssembledPencil &p1_assembly
    ) {
        auto multigrid_future = std::async(std::launch::async, [&] {
            return modal::finite_cell::MetalOperations::PrepareP1Multigrid(fem, alpha, p1_assembly);
        });
        auto metal_future = std::async(std::launch::async, [&] { return std::make_unique<modal::finite_cell::MetalOperations>(fem); });
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
modal::FiniteCellEigenpairs SolveFactorFreeMetal(
    const modal::FiniteCellOperator &fem, uint32_t count, double alpha, double tolerance,
    uint32_t max_iterations
) {
    max_iterations = std::min(100u, max_iterations);
    constexpr uint32_t guard_vectors{4}, stagnation_window{12};
    constexpr double stagnation_reduction{0.8};
    const auto start = Clock::now();
    modal::FiniteCellEigenpairs result;
    Actions actions{fem};
    const auto apply_mass_shifted = [&](const double *input, double *mass, double *shifted, uint32_t width) {
        actions.ApplyMassShifted(input, mass, shifted, width, alpha);
    };
    double preconditioner_setup_seconds{}, preconditioner_seconds{}, ritz_seconds{};
    double initialization_seconds{}, residual_seconds{}, recurrence_seconds{};
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
    uint32_t width = std::min<uint32_t>(fem.Dofs(), count + std::min(guard_vectors, count));
    numeric::Matrix<double> space = InitialSpace(fem, width, alpha, *p1_assembly);
    initialization_seconds += SecondsSince(initialization_start);
    numeric::Matrix<double> mass_space, vectors, mass_vectors, shifted_vectors;
    numeric::Vector<double> shifted_values;
    if (!Ritz(actions, alpha, space, mass_space, width, vectors, mass_vectors, shifted_vectors, shifted_values, ritz_seconds)) {
        result.Iterations = std::numeric_limits<uint32_t>::max();
        return finish();
    }
    const auto relative_residual = [alpha](numeric::VectorView<const double> mass, numeric::VectorView<const double> shifted, numeric::VectorView<const double> residual, double shifted_value) {
        double stiffness_squared{};
        for (size_t row = 0; row < mass.size(); ++row) stiffness_squared += std::pow(shifted[row] - alpha * mass[row], 2);
        const double scale = std::sqrt(stiffness_squared) + std::abs(shifted_value - alpha) * numeric::Norm(mass);
        const double residual_norm = numeric::Norm(residual);
        return scale == 0 ? residual_norm : residual_norm / scale;
    };
    uint32_t locked_count = count > 6 ? 6 : 0;
    numeric::Matrix<double> locked_vectors, locked_mass_vectors;
    numeric::Vector<double> locked_values, locked_relative;
    if (locked_count) {
        locked_vectors.Resize(fem.Dofs(), requested_count);
        locked_mass_vectors.Resize(fem.Dofs(), requested_count);
        locked_values.Resize(requested_count);
        locked_relative.Resize(requested_count);
        numeric::Copy(vectors.FirstColumns(locked_count), locked_vectors.FirstColumns(locked_count));
        numeric::Copy(mass_vectors.FirstColumns(locked_count), locked_mass_vectors.FirstColumns(locked_count));
        numeric::Copy(shifted_values.First(locked_count), locked_values.First(locked_count));
        for (uint32_t mode = 0; mode < locked_count; ++mode) {
            numeric::Vector<double> residual(fem.Dofs());
            numeric::Copy(shifted_vectors.Column(mode), residual.View());
            numeric::AddScaled(-shifted_values[mode], mass_vectors.Column(mode), residual.View());
            locked_relative[mode] = relative_residual(mass_vectors.Column(mode), shifted_vectors.Column(mode), residual.View(), shifted_values[mode]);
        }
        RemoveLeadingColumns(vectors, locked_count);
        RemoveLeadingColumns(mass_vectors, locked_count);
        RemoveLeadingColumns(shifted_vectors, locked_count);
        RemoveLeadingEntries(shifted_values, locked_count);
        count -= locked_count;
        width -= locked_count;
    }
    // The six rigid-body eigenvalues equal the shift within the configured relative threshold.
    const auto rigid = [&](uint32_t mode) { return std::abs(shifted_values[mode] - alpha) <= alpha * 1e-4; };
    const auto accept = [&](const numeric::Vector<double> &active_relative) {
        result.Eigenvalues.Resize(requested_count);
        result.Eigenvectors.Resize(fem.Dofs(), requested_count);
        result.RelativeResiduals.Resize(requested_count);
        if (locked_count) {
            for (uint32_t mode = 0; mode < locked_count; ++mode) result.Eigenvalues[mode] = locked_values[mode] - alpha;
            numeric::Copy(locked_vectors.FirstColumns(locked_count), result.Eigenvectors.FirstColumns(locked_count));
            numeric::Copy(locked_relative.First(locked_count), result.RelativeResiduals.First(locked_count));
        }
        for (uint32_t mode = 0; mode < count; ++mode) result.Eigenvalues[locked_count + mode] = shifted_values[mode] - alpha;
        numeric::Copy(vectors.FirstColumns(count), result.Eigenvectors.LastColumns(count));
        numeric::Copy(active_relative.First(count), result.RelativeResiduals.Last(count));
    };
    const auto wait_start = Clock::now();
    auto preconditioner = preconditioner_setup_future.get();
    preconditioner_setup_seconds = SecondsSince(wait_start);
    p1_assembly.reset();
    actions.PackedCut = std::move(preconditioner.CutActions);
    numeric::Matrix<float> compact_previous_direction;
    numeric::Matrix<double> previous_mass_direction, previous_shifted_direction;
    std::future<double> previous_actions;
    const auto wait_previous_actions = [&] {
        if (!previous_actions.valid()) return;
        const auto wait_start = Clock::now();
        previous_actions.get();
        actions.Seconds += SecondsSince(wait_start);
    };
    std::vector<double> stagnation_history;

    for (uint32_t iteration = 0; iteration < max_iterations; ++iteration) {
        const auto residual_start = Clock::now();
        const double residual_action_start = actions.Seconds;
        numeric::Matrix<double> residual = numeric::ColumnScaledDifference(shifted_vectors.View(), mass_vectors.View(), shifted_values.View());
        numeric::Vector<double> relative(width);
        const auto update_relative = [&] {
            for (uint32_t i = 0; i < width; ++i)
                relative[i] = relative_residual(
                    mass_vectors.Column(i), shifted_vectors.Column(i), residual.Column(i), shifted_values[i]
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
            wait_previous_actions();
            // Recompute the residual from exact actions before accepting a recurrence-driven convergence.
            actions.ApplyMassShifted(vectors.data(), mass_vectors.data(), shifted_vectors.data(), width, alpha);
            residual = numeric::ColumnScaledDifference(shifted_vectors.View(), mass_vectors.View(), shifted_values.View());
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
        if (newly_locked) {
            wait_previous_actions();
            numeric::Matrix<double> exact_mass(fem.Dofs(), newly_locked), exact_shifted(fem.Dofs(), newly_locked);
            actions.ApplyMassShifted(vectors.data(), exact_mass.data(), exact_shifted.data(), newly_locked, alpha);
            numeric::Copy(exact_mass.View(), mass_vectors.FirstColumns(newly_locked));
            numeric::Copy(exact_shifted.View(), shifted_vectors.FirstColumns(newly_locked));
            residual = numeric::ColumnScaledDifference(shifted_vectors.View(), mass_vectors.View(), shifted_values.View());
            update_relative();
            newly_locked = 0;
            while (newly_locked < count && relative[newly_locked] < tolerance) ++newly_locked;
        }
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
                    wait_previous_actions();
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
            wait_previous_actions();
            const uint32_t old_locked = locked_count;
            locked_count += newly_locked;
            numeric::Copy(vectors.FirstColumns(newly_locked), locked_vectors.ColumnsAt(old_locked, newly_locked));
            numeric::Copy(mass_vectors.FirstColumns(newly_locked), locked_mass_vectors.ColumnsAt(old_locked, newly_locked));
            numeric::Copy(shifted_values.First(newly_locked), locked_values.Subvector(old_locked, newly_locked));
            numeric::Copy(relative.First(newly_locked), locked_relative.Subvector(old_locked, newly_locked));
            RemoveLeadingColumns(vectors, newly_locked);
            RemoveLeadingColumns(mass_vectors, newly_locked);
            RemoveLeadingColumns(shifted_vectors, newly_locked);
            RemoveLeadingEntries(shifted_values, newly_locked);
            RemoveLeadingColumns(residual, newly_locked);
            RemoveLeadingEntries(relative, newly_locked);
            count -= newly_locked;
            width -= newly_locked;
            compact_previous_direction.Clear();
            previous_mass_direction.Clear();
            previous_shifted_direction.Clear();
        }

        std::vector<uint32_t> active;
        active.reserve(width);
        for (uint32_t i = 0; i < width; ++i)
            if (i >= count || (!rigid(i) && relative[i] >= tolerance)) active.push_back(i);
        numeric::Matrix<double> active_residual(fem.Dofs(), active.size());
        for (uint32_t i = 0; i < active.size(); ++i) numeric::Copy(residual.Column(active[i]), active_residual.Column(i));
        residual.Clear();

        const size_t correction_columns = active.size();
        const size_t history_columns = compact_previous_direction.cols();
        space.Resize(fem.Dofs(), width + correction_columns + history_columns);
        mass_space.Resize(space.rows(), space.cols());
        numeric::Matrix<double> shifted_space(fem.Dofs(), space.cols());
        numeric::Copy(vectors.View(), space.FirstColumns(width));
        numeric::Copy(mass_vectors.View(), mass_space.FirstColumns(width));
        numeric::Copy(shifted_vectors.View(), shifted_space.FirstColumns(width));
        const auto preconditioner_start = Clock::now();
        preconditioner.Apply(
            active_residual.data(), space.ColumnsAt(width, correction_columns).data(), uint32_t(active.size())
        );
        preconditioner_seconds += SecondsSince(preconditioner_start);
        wait_previous_actions();
        apply_mass_shifted(
            space.ColumnsAt(width, correction_columns).data(), mass_space.ColumnsAt(width, correction_columns).data(),
            shifted_space.ColumnsAt(width, correction_columns).data(), uint32_t(correction_columns)
        );
        active_residual.Clear();
        if (history_columns) {
            const auto history = numeric::Cast<double>(compact_previous_direction.View());
            numeric::Copy(history.View(), space.LastColumns(history_columns));
            numeric::Copy(previous_mass_direction.View(), mass_space.LastColumns(history_columns));
            numeric::Copy(previous_shifted_direction.View(), shifted_space.LastColumns(history_columns));
        }
        if (!RitzFromActions(
                space, mass_space, shifted_space, width, vectors, mass_vectors, shifted_vectors,
                shifted_values, ritz_seconds, locked_vectors, locked_mass_vectors, locked_values, locked_count
            )) {
            result.Iterations = iteration + 1;
            return finish();
        }
        compact_previous_direction.Resize(fem.Dofs(), active.size());
        previous_mass_direction.Resize(fem.Dofs(), active.size());
        previous_shifted_direction.Resize(fem.Dofs(), active.size());
        for (uint32_t i = 0; i < active.size(); ++i) {
            const auto vector = vectors.Column(active[i]);
            const numeric::MatrixView<const double> vector_matrix{vector.Values, vector.Count, 1, vector.Count};
            const numeric::Matrix<double> coefficients = numeric::TransposeMultiply(mass_space.FirstColumns(width), vector_matrix);
            const numeric::Matrix<double> projection = numeric::Multiply(space.FirstColumns(width), coefficients.View());
            for (size_t row = 0; row < vector.Count; ++row) compact_previous_direction(row, i) = float(vector[row] - projection(row, 0));
        }
        previous_actions = std::async(std::launch::async, [&] {
            const numeric::Matrix<double> direction = numeric::Cast<double>(compact_previous_direction.View());
            return actions.ApplyMassShiftedTimed(
                direction.data(), previous_mass_direction.data(), previous_shifted_direction.data(),
                uint32_t(direction.cols()), alpha
            );
        });

        recurrence_seconds += SecondsSince(recurrence_start) - (actions.Seconds - recurrence_action_start) -
            (preconditioner_seconds - recurrence_preconditioner_start) - (ritz_seconds - recurrence_ritz_start);
        result.Iterations = iteration + 1;
    }

    wait_previous_actions();
    const auto residual_start = Clock::now();
    const numeric::Matrix<double> residual = numeric::ColumnScaledDifference(shifted_vectors.FirstColumns(count), mass_vectors.FirstColumns(count), shifted_values.First(count));
    numeric::Vector<double> relative(count);
    for (uint32_t i = 0; i < count; ++i)
        relative[i] = relative_residual(
            mass_vectors.Column(i), shifted_vectors.Column(i), residual.Column(i), shifted_values[i]
        );
    residual_seconds += SecondsSince(residual_start);
    accept(relative);
    return finish();
}

} // namespace

modal::FiniteCellEigenpairs modal::SolveFiniteCellEigenpairs(
    const FiniteCellOperator &fem, uint32_t count, double alpha, double tolerance, uint32_t max_iterations
) {
    const auto start = Clock::now();
    auto factor_free = SolveFactorFreeMetal(fem, count, alpha, tolerance, max_iterations);
    const uint32_t first_physical = std::min(6u, count), physical_count = count - first_physical;
    const auto converged = [&](const FiniteCellEigenpairs &result) {
        return result.Eigenvalues.size() == count && result.RelativeResiduals.size() == count && numeric::AllFinite(result.Eigenvalues.View()) && numeric::AllFinite(result.RelativeResiduals.View()) &&
            (!physical_count || numeric::Maximum(result.RelativeResiduals.Subvector(first_physical, physical_count)) < tolerance);
    };
    if (converged(factor_free)) return factor_free;
    const double attempt_seconds = SecondsSince(start);
    const uint32_t attempt_iterations = factor_free.Iterations;
    const bool attempt_stagnated = factor_free.Profile.Stagnated;
    const bool attempt_converged = factor_free.RelativeResiduals.size() == count;
    const double attempt_residual = attempt_converged && physical_count ?
        numeric::Maximum(factor_free.RelativeResiduals.Subvector(first_physical, physical_count)) :
        physical_count ? std::numeric_limits<double>::infinity() :
                         0;
    factor_free = {};
    auto assembled = finite_cell::SolveAssembledEigenpairs(fem, count, alpha, tolerance, max_iterations);
    if (!converged(assembled)) throw std::runtime_error("Finite-cell default and fallback solvers did not converge within the iteration budget.");
    assembled.Profile.FailedFactorFreeSeconds = attempt_seconds;
    assembled.Profile.FailedFactorFreeIterations = attempt_iterations;
    assembled.Profile.FailedFactorFreeStagnated = attempt_stagnated;
    assembled.Profile.FailedFactorFreeResidual = attempt_residual;
    assembled.Profile.Total += attempt_seconds;
    return assembled;
}

modal::FiniteCellEigenpairs modal::finite_cell::SolveAssembledEigenpairs(
    const FiniteCellOperator &fem, uint32_t count, double alpha, double tolerance,
    uint32_t max_iterations
) {
    const auto start = Clock::now();
    FiniteCellEigenpairs result;
    if (!count || count >= fem.Dofs()) return result;
    const auto assembly_start = Clock::now();
    const auto assembled = fem.AssembleLower();
    result.Profile.Actions = SecondsSince(assembly_start);
    if (fem.Dofs() <= 2048 && count * 8 >= fem.Dofs()) {
        const auto dense_start = Clock::now();
        numeric::Matrix<double> dense_stiffness = assembled.Stiffness.DenseSymmetric();
        numeric::Matrix<double> dense_mass = assembled.Mass.DenseSymmetric();
        numeric::Vector<double> eigenvalues(fem.Dofs());
        if (numeric::GeneralizedSelfAdjointEigenSolve(
                dense_stiffness.data(), dense_mass.data(), eigenvalues.data(), fem.Dofs()
            )) {
            result.Profile.RayleighRitz = SecondsSince(dense_start);
            result.Eigenvalues = numeric::Copy(eigenvalues.First(count));
            result.Eigenvectors = numeric::Copy(dense_stiffness.FirstColumns(count));
            const auto residual_start = Clock::now();
            const numeric::Matrix<double> mass_vectors = numeric::SymmetricMultiply(assembled.Mass, result.Eigenvectors.View());
            const numeric::Matrix<double> stiffness_vectors = numeric::SymmetricMultiply(assembled.Stiffness, result.Eigenvectors.View());
            SetResiduals(result, mass_vectors, stiffness_vectors, 0);
            result.Profile.Residuals = SecondsSince(residual_start);
            result.Iterations = 1;
            result.Profile.Total = SecondsSince(start);
            result.Profile.Other = result.Profile.Total - result.Profile.Actions - result.Profile.RayleighRitz - result.Profile.Residuals;
            return result;
        }
    }
    double factor_seconds{}, solve_seconds{};
    modal::finite_cell::AccelerateShiftInvert inverse{assembled.Stiffness, assembled.Mass, factor_seconds, solve_seconds};
    const uint32_t solve_count = std::min<uint32_t>(fem.Dofs() - 1, count + 4);
    const uint32_t basis = std::min<uint32_t>(fem.Dofs(), solve_count + 20);
    const auto eigensolver = modal::eigensolver::SolveGeneralizedEigenproblem(
        inverse, assembled.Mass, assembled.Stiffness,
        {
            .Count = solve_count,
            .CertifiedCount = count,
            .SubspaceSize = basis,
            .Shift = -alpha,
            .IterationTolerance = 1e-5,
            .ResidualTolerance = 0.25 * tolerance,
            .MaxIterations = max_iterations,
            .MaxRefinementIterations = 50,
            .RandomSeed = 20260828,
        }
    );
    result.Iterations = eigensolver.Iterations;
    if (eigensolver.Converged) {
        numeric::Matrix<double> space = eigensolver.Eigenvectors;
        result.Eigenvalues = numeric::Copy(eigensolver.Eigenvalues.First(count));
        result.Eigenvectors = numeric::Copy(space.FirstColumns(count));
        result.RelativeResiduals = numeric::Copy(eigensolver.RelativeResiduals.First(count));
        const uint32_t first_physical = std::min(6u, count), physical_count = count - first_physical;
        const bool safely_converged = eigensolver.MassOrthogonalityError < 0.5e-9 &&
            (!physical_count || numeric::Maximum(result.RelativeResiduals.Subvector(first_physical, physical_count)) < 0.5 * tolerance);
        if (!safely_converged) {
            numeric::Matrix<double> mass_space, vectors, mass_vectors, shifted_vectors;
            numeric::Vector<double> shifted_values;
            Actions actions{fem};
            if (Ritz(
                    actions, alpha, space, mass_space, solve_count, vectors, mass_vectors,
                    shifted_vectors, shifted_values, result.Profile.RayleighRitz
                )) {
                numeric::Matrix<double> stiffness_vectors = numeric::Copy(shifted_vectors.View());
                numeric::AddScaled(-alpha, mass_vectors.View(), stiffness_vectors.View());
                const numeric::Matrix<double> residual = numeric::ColumnScaledDifference(shifted_vectors.View(), mass_vectors.View(), shifted_values.View());
                std::vector<uint32_t> active;
                for (uint32_t mode = first_physical; mode < count; ++mode) {
                    const double scale = numeric::Norm(stiffness_vectors.Column(mode)) +
                        std::abs(shifted_values[mode] - alpha) * numeric::Norm(mass_vectors.Column(mode));
                    const double residual_norm = numeric::Norm(residual.Column(mode));
                    const double relative = scale == 0 ? residual_norm : residual_norm / scale;
                    if (relative >= 0.5 * tolerance) active.push_back(mode);
                }
                numeric::Matrix<double> refined_vectors;
                numeric::Vector<double> refined_values;
                bool refined = active.empty();
                if (refined) {
                    refined_vectors = numeric::Copy(vectors.FirstColumns(count));
                    refined_values = numeric::Copy(shifted_values.First(count));
                    for (double &value : refined_values.Values) value -= alpha;
                } else {
                    numeric::Matrix<double> active_residual(fem.Dofs(), active.size());
                    for (uint32_t column = 0; column < active.size(); ++column)
                        numeric::Copy(residual.Column(active[column]), active_residual.Column(column));
                    numeric::Matrix<double> correction(fem.Dofs(), active.size());
                    inverse.solve_panel(active_residual.data(), correction.data(), int(active.size()));
                    space.Resize(fem.Dofs(), solve_count + active.size());
                    numeric::Copy(vectors.View(), space.FirstColumns(solve_count));
                    numeric::Copy(correction.View(), space.LastColumns(active.size()));
                    refined = Ritz(
                        actions, alpha, space, mass_space, count, refined_vectors, mass_vectors,
                        shifted_vectors, shifted_values, result.Profile.RayleighRitz
                    );
                    if (refined) {
                        refined_values = shifted_values;
                        for (double &value : refined_values.Values) value -= alpha;
                    }
                }
                if (refined) {
                    result.Eigenvalues = std::move(refined_values);
                    result.Eigenvectors = std::move(refined_vectors);
                    SetResiduals(result, mass_vectors, shifted_vectors, alpha);
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
