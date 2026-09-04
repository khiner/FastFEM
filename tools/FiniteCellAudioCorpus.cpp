#include "FiniteCellBenchmarkGeometry.h"
#include "ModeShapeComparison.h"
#include "audio/FiniteCell.h"
#include "audio/FiniteCellEigensolver.h"
#include "audio/finite_cell/AssembledCholesky.h"

#include <charconv>
#include <cmath>
#include <numbers>
#include <print>
#include <stdexcept>
#include <string_view>

namespace {
constexpr AcousticMaterialProperties Material{.Density = 2700, .YoungModulus = 7.2e10, .PoissonRatio = 0.19};

uint32_t Parse(std::string_view value) {
    uint32_t result{};
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size() || result == 0)
        throw std::invalid_argument("Audio-corpus resolution and mode count must be positive integers.");
    return result;
}

bool Run(std::string_view name, uint32_t longitudinal, uint32_t count) {
    constexpr double tolerance = 1e-8;
    const double shift = std::pow(2 * std::numbers::pi * 20, 2);
    const auto geometry = finite_cell_benchmark::MakeGeometry(name);
    const auto domain = modal::MakeTriangleSurfaceDomain(geometry.Boundary.Points, geometry.Boundary.Triangles);
    const uvec3 cells = finite_cell_benchmark::GridResolution(geometry, longitudinal);
    const auto operation = modal::BuildFiniteCellOperator(
        domain, Material,
        {.Cells = cells, .CutDepth = 2, .FictitiousScale = 1e-8, .PaddingCells = 0.25}
    );
    if (count + 4 >= operation.Dofs())
        throw std::invalid_argument("Audio-corpus mode count leaves no guard-vector space for " + std::string{name} + ".");
    const auto assembled = modal::finite_cell::SolveAssembledCholesky(operation, count, shift, 1e-9, 1000);
    const uint32_t max_iterations = std::max(200u, 2 * count);
    const auto result = modal::SolveFiniteCellEigenpairs(operation, count, shift, tolerance, max_iterations);
    if (assembled.Eigenvalues.size() != count || result.Eigenvalues.size() != count) {
        std::println(stderr, "audio geometry={} modes={}/{} did not converge", name, assembled.Eigenvalues.size(), result.Eigenvalues.size());
        return false;
    }
    const double spectrum_error = (result.Eigenvalues.tail(count - 6) - assembled.Eigenvalues.tail(count - 6)).norm() /
        assembled.Eigenvalues.tail(count - 6).norm();
    const auto result_certification = modal::CertifyFiniteCellEigenpairs(operation, result.Eigenvalues, result.Eigenvectors);
    Eigen::Index worst_physical{};
    const double residual = result_certification.RelativeResiduals.tail(count - 6).maxCoeff(&worst_physical);
    const double recurrence_residual = result.RelativeResiduals.tail(count - 6).maxCoeff();
    const uint32_t worst_mode = uint32_t(worst_physical) + 6;
    const uint32_t uncertified = uint32_t((result_certification.RelativeResiduals.tail(count - 6).array() >= tolerance).count());
    const auto shapes = finite_cell_benchmark::CompareSameDiscretizationModeShapes(
        operation, assembled.Eigenvalues, assembled.Eigenvectors, result.Eigenvectors
    );
    const double first_frequency = std::sqrt(std::max(0.0, result.Eigenvalues[6])) / (2 * std::numbers::pi);
    const double last_frequency = std::sqrt(std::max(0.0, result.Eigenvalues[count - 1])) / (2 * std::numbers::pi);
    std::println(
        "audio geometry={} description={} grid={}x{}x{} dofs={} cut={} modes={} iterations={}/{} failed_factor_free={}/{}/{:.3e} assembled_iterations={} frequency={:.3f}/{:.3f} spectrum={:.3e} residual={:.3e}/{:.3e}@{} uncertified={} orthogonality={:.3e} paired_mac={:.6f} best_mac={:.6f} cluster_mac={:.6f} clusters={}/{}",
        name, geometry.Description, cells.x, cells.y, cells.z,
        operation.Dofs(), operation.Profile.CutCells, count, result.Iterations, max_iterations,
        result.Profile.FailedFactorFreeIterations, result.Profile.FailedFactorFreeStagnated,
        result.Profile.FailedFactorFreeResidual,
        assembled.Iterations, first_frequency, last_frequency,
        spectrum_error, recurrence_residual, residual, worst_mode, uncertified, result_certification.MassOrthogonalityError, shapes.PairedMacMinimum,
        shapes.BestMacMinimum, shapes.ClusterMacMinimum, shapes.Clusters, shapes.LargestCluster
    );
    if (spectrum_error >= 1e-9)
        for (uint32_t mode = 6; mode < count; ++mode) {
            const double assembled_frequency = std::sqrt(std::max(0.0, assembled.Eigenvalues[mode])) / (2 * std::numbers::pi);
            const double result_frequency = std::sqrt(std::max(0.0, result.Eigenvalues[mode])) / (2 * std::numbers::pi);
            const double relative = std::abs(result.Eigenvalues[mode] - assembled.Eigenvalues[mode]) /
                std::max(std::abs(assembled.Eigenvalues[mode]), 1.0);
            if (relative > 1e-6)
                std::println(stderr, "audio mismatch geometry={} mode={} assembled={:.9f}/{:.6e} result={:.9f}/{:.6e} relative={:.3e}", name, mode, assembled_frequency, assembled.Eigenvalues[mode], result_frequency, result.Eigenvalues[mode], relative);
        }
    return spectrum_error < 1e-9 && residual < tolerance &&
        result_certification.MassOrthogonalityError < 1e-9 && shapes.ClusterMacMinimum > 0.99999;
}
} // namespace

int main(int argc, char **argv) try {
    if (argc < 3 || argc > 4) {
        std::println(stderr, "usage: {} longitudinal-cells mode-count [audio-geometry|all]", argv[0]);
        return 2;
    }
    const uint32_t longitudinal = Parse(argv[1]), count = Parse(argv[2]);
    if (count <= 6) throw std::invalid_argument("Audio-corpus mode count must include at least one physical mode after the six rigid modes.");
    const std::string_view selected = argc == 4 ? argv[3] : "all";
    if (selected != "all" && !finite_cell_benchmark::IsGeometry(selected))
        throw std::invalid_argument("Unknown finite-cell geometry: " + std::string{selected});
    bool passed = true;
    if (selected == "all")
        for (const std::string_view name : finite_cell_benchmark::AudioGeometryNames) passed &= Run(name, longitudinal, count);
    else
        passed = Run(selected, longitudinal, count);
    return passed ? 0 : 1;
} catch (const std::exception &error) {
    std::println(stderr, "{}", error.what());
    return 2;
}
