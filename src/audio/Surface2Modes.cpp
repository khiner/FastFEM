#include "Surface2Modes.h"

#include "FiniteCellEigensolver.h"
#include "MassPropertiesAccumulator.h"
#include "ModalResultBuilder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <numbers>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

template<typename Operation> auto Timed(double &seconds, Operation &&operation) {
    const auto start = Clock::now();
    auto result = operation();
    seconds += std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}

MassProperties ComputeFiniteCellMassProperties(const modal::FiniteCellOperator &operation, double density, vec3 baked_scale, double length_to_si) {
    const dvec3 inverse_scale{1.0 / baked_scale.x, 1.0 / baked_scale.y, 1.0 / baked_scale.z};
    modal::MassPropertiesAccumulator accumulator;
    for (const auto &cell : operation.Cells) {
        const dvec3 half = 1.0 / cell.InverseHalf;
        const dvec3 center = operation.Nodes[cell.Nodes[0]] + half;
        for (uint32_t offset = 0; offset < cell.QuadratureCount; ++offset) {
            const auto &point = operation.Quadrature[cell.QuadratureOffset + offset];
            if (point.Fictitious) continue;
            accumulator.Add((center + point.Reference * half) * inverse_scale, point.Weight);
        }
    }
    return accumulator.Finish(density, length_to_si);
}

std::expected<modal::ModalResult, std::string> SolveFiniteCell(std::span<const vec3> surface_positions, std::span<const uint32_t> triangle_indices, const AcousticMaterialProperties &material, std::span<const vec3> excitation_positions, vec3 baked_scale, const modal::SurfaceSolveConfig &config, modal::SolveReuse reuse, modal::SolveMonitor *monitor) {
    if (reuse.SeedBasis || reuse.Cache) return std::unexpected("Finite-cell surface solves do not accept a Tet10 seed basis or solve cache.");
    if (config.Modal.NumFemModes == 0) return std::unexpected("A surface solve requires at least one FEM mode.");
    std::vector<dvec3> positions;
    positions.reserve(surface_positions.size());
    for (const auto point : surface_positions) positions.emplace_back(point);

    modal::SolveProfile profile;
    const auto domain = modal::MakeTriangleSurfaceDomain(positions, triangle_indices);
    if (monitor) monitor->Progress.store(0.1f, std::memory_order_relaxed);
    auto operation = Timed(profile.Assemble, [&] { return modal::BuildFiniteCellOperator(domain, material, config.FiniteCell); });
    profile.Dofs = operation.Dofs();
    if (monitor && monitor->Cancelled()) return modal::ModalResult{};
    if (operation.Dofs() < 2) return std::unexpected("The finite-cell grid has insufficient degrees of freedom.");

    const uint32_t count = std::min(config.Modal.NumFemModes, operation.Dofs() - 1);
    const double alpha = std::pow(2 * std::numbers::pi * config.Modal.MinModeFreq, 2);
    if (monitor) monitor->Progress.store(0.3f, std::memory_order_relaxed);
    auto eigenpairs = Timed(profile.Iterate, [&] { return modal::SolveFiniteCellEigenpairs(operation, count, alpha, config.Modal.Tolerance, config.Modal.MaxRestarts); });
    profile.Factorize = eigenpairs.Profile.PreconditionerSetup;
    profile.Iterate -= profile.Factorize;
    profile.OpSolve = eigenpairs.Profile.Preconditioner;
    profile.Restarts = eigenpairs.Iterations;
    if (eigenpairs.Eigenvalues.size() != count || eigenpairs.Eigenvectors.cols() != count) return std::unexpected("The finite-cell eigensolver did not return the requested modes.");
    if (eigenpairs.RelativeResiduals.size() == count && count > 6) profile.PhysicalResidual = numeric::Maximum(eigenpairs.RelativeResiduals.Last(count - 6));
    if (monitor) monitor->Progress.store(0.95f, std::memory_order_relaxed);
    if (monitor && monitor->Cancelled()) return modal::ModalResult{};

    const dvec3 inverse_scale{1.0 / baked_scale.x, 1.0 / baked_scale.y, 1.0 / baked_scale.z};
    std::vector<std::vector<vec3>> shapes;
    std::vector<vec3> sample_positions;
    shapes.reserve(excitation_positions.size());
    sample_positions.reserve(excitation_positions.size());
    const auto sample_start = Clock::now();
    for (const auto position : excitation_positions) {
        const auto stencil = operation.InterpolationAt(dvec3{position});
        if (!stencil) return std::unexpected("An excitation position lies outside the active finite-cell grid.");
        auto &shape = shapes.emplace_back(eigenpairs.Eigenvalues.size());
        for (size_t mode = 0; mode < eigenpairs.Eigenvalues.size(); ++mode)
            for (uint32_t node = 0; node < stencil->Count; ++node)
                for (uint32_t component = 0; component < 3; ++component)
                    shape[size_t(mode)][component] += float(stencil->Weights[node] * eigenpairs.Eigenvectors(3 * stencil->Nodes[node] + component, mode));
        sample_positions.emplace_back(dvec3{position} * inverse_scale);
    }
    profile.SampleExcite = std::chrono::duration<double>(Clock::now() - sample_start).count();

    const double length_to_si = (double(baked_scale.x) + baked_scale.y + baked_scale.z) / 3;
    auto mass_properties = Timed(profile.MassProps, [&] { return ComputeFiniteCellMassProperties(operation, material.Density, baked_scale, length_to_si); });
    numeric::Matrix<float> basis;
    if (reuse.KeepBasis) basis = numeric::Cast<float>(eigenpairs.Eigenvectors.View());
    std::vector<uint32_t> sample_point_of(excitation_positions.size());
    for (uint32_t point = 0; point < sample_point_of.size(); ++point) sample_point_of[point] = point;
    if (monitor) monitor->Progress.store(1, std::memory_order_relaxed);
    return modal::BuildModalResult({eigenpairs.Eigenvalues.begin(), eigenpairs.Eigenvalues.end()}, std::move(shapes), material, config.Modal, std::move(sample_positions), baked_scale, std::move(mass_properties), profile, std::move(basis), std::move(sample_point_of));
}
} // namespace

std::expected<modal::ModalResult, std::string> modal::Surface2Modes(std::span<const vec3> positions, std::span<const uint32_t> triangle_indices, const AcousticMaterialProperties &material, std::span<const vec3> excitation_positions, vec3 baked_scale, Discretization discretization, SurfaceSolveConfig config, SolveReuse reuse, SolveMonitor *monitor) {
    if (positions.empty() || triangle_indices.empty() || triangle_indices.size() % 3) return std::unexpected("A surface solve requires indexed triangles.");
    if (baked_scale.x <= 0 || baked_scale.y <= 0 || baked_scale.z <= 0) return std::unexpected("Baked scale components must be positive.");
    try {
        switch (discretization) {
            case Discretization::Tet10: {
                auto tetrahedra = GenerateTets({positions.begin(), positions.end()}, {triangle_indices.begin(), triangle_indices.end()}, config.Tetrahedralization);
                if (!tetrahedra) return std::unexpected(std::move(tetrahedra.error()));
                return SolveTet10Modes(tetrahedra->Mesh, material, {excitation_positions.begin(), excitation_positions.end()}, baked_scale, config.Modal, reuse, monitor);
            }
            case Discretization::FiniteCell:
                return SolveFiniteCell(positions, triangle_indices, material, excitation_positions, baked_scale, config, reuse, monitor);
        }
    } catch (const std::exception &error) {
        return std::unexpected(error.what());
    }
    return std::unexpected("Unknown surface discretization.");
}
