#include "FiniteCellBenchmarkGeometry.h"
#include "LoadObj.h"
#include "ModeShapeComparison.h"
#include "audio/FiniteCell.h"
#include "audio/FiniteCellEigensolver.h"
#include "audio/finite_cell/AssembledCholesky.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <print>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {
constexpr uint32_t AcceptedModeCount{18}, ShapeGuardModes{8}, SolvedModeCount{AcceptedModeCount + ShapeGuardModes};
constexpr double Omega{2 * std::numbers::pi * 20};
constexpr double Shift{Omega * Omega};
constexpr AcousticMaterialProperties Material{.Density = 2700, .YoungModulus = 7.2e10, .PoissonRatio = 0.19};

struct Geometry {
    finite_cell_benchmark::Surface Boundary;
    dvec3 Extent;
};

struct CellFractions {
    double Minimum{}, Median{};
    uint32_t BelowOnePercent{};
};

struct RunResult {
    bool Passed{}, FellBack{}, Stagnated{};
};

uint32_t Parse(std::string_view text, std::string_view description) {
    uint32_t result{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
    if (error != std::errc{} || end != text.data() + text.size() || !result)
        throw std::invalid_argument(std::string{description} + " must be a positive integer.");
    return result;
}

bool IsWatertight(std::span<const uint32_t> triangles) {
    if (triangles.empty() || triangles.size() % 3) return false;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> edges;
    for (size_t triangle = 0; triangle < triangles.size(); triangle += 3) {
        const uint32_t a = triangles[triangle], b = triangles[triangle + 1], c = triangles[triangle + 2];
        if (a == b || b == c || c == a) return false;
        ++edges[std::minmax(a, b)];
        ++edges[std::minmax(b, c)];
        ++edges[std::minmax(c, a)];
    }
    return std::ranges::all_of(edges, [](const auto &edge) { return edge.second == 2; });
}

Geometry LoadGeometry(const fs::path &path) {
    const auto loaded = LoadObj(path);
    if (!loaded || loaded->Positions.empty() || !IsWatertight(loaded->TriangleIndices))
        throw std::invalid_argument("not a nonempty watertight triangle mesh");
    finite_cell_benchmark::Surface surface;
    surface.Triangles = loaded->TriangleIndices;
    surface.Points.reserve(loaded->Positions.size());
    dvec3 minimum{std::numeric_limits<double>::max()}, maximum{std::numeric_limits<double>::lowest()};
    for (const vec3 &point : loaded->Positions) {
        const dvec3 value{point};
        minimum = numeric::Min(minimum, value);
        maximum = numeric::Max(maximum, value);
    }
    const dvec3 original_extent = maximum - minimum;
    const double longest = std::max({original_extent.x, original_extent.y, original_extent.z});
    if (!(longest > 0) || !std::isfinite(longest)) throw std::invalid_argument("has invalid bounds");
    const double scale = 0.3 / longest;
    for (const vec3 &point : loaded->Positions) surface.Points.push_back((dvec3{point} - minimum) * scale);
    const auto [normalized_minimum, normalized_maximum] = finite_cell_benchmark::Bounds(surface);
    return {.Boundary = std::move(surface), .Extent = normalized_maximum - normalized_minimum};
}

CellFractions PhysicalCellFractions(const modal::FiniteCellOperator &operation) {
    std::vector<double> fractions;
    fractions.reserve(operation.Cells.size());
    for (const auto &cell : operation.Cells) {
        double physical{};
        for (uint32_t point = cell.QuadratureOffset; point < cell.QuadratureOffset + cell.QuadratureCount; ++point)
            if (!operation.Quadrature[point].Fictitious) physical += operation.Quadrature[point].Weight;
        fractions.push_back(physical * cell.InverseHalf.x * cell.InverseHalf.y * cell.InverseHalf.z / 8);
    }
    std::ranges::sort(fractions);
    return {
        .Minimum = fractions.front(),
        .Median = fractions[fractions.size() / 2],
        .BelowOnePercent = uint32_t(std::ranges::count_if(fractions, [](double fraction) { return fraction < 0.01; })),
    };
}

std::vector<fs::path> Objects(const fs::path &root, const std::vector<std::string> &datasets, const std::set<std::string> &names) {
    std::vector<fs::path> result;
    const auto collect = [&](const fs::path &directory) {
        if (!fs::is_directory(directory)) throw std::invalid_argument("corpus dataset is absent: " + directory.string());
        for (const auto &entry : fs::directory_iterator{directory})
            if (entry.is_regular_file() && entry.path().extension() == ".obj" && (names.empty() || names.contains(entry.path().stem().string())))
                result.push_back(entry.path());
    };
    if (datasets.empty())
        for (const auto &entry : fs::directory_iterator{root})
            if (entry.is_directory()) collect(entry.path());
    for (const auto &dataset : datasets) collect(root / dataset);
    std::ranges::sort(result);
    return result;
}

RunResult Run(const fs::path &path, uint32_t resolution, uint32_t max_iterations) {
    const auto geometry = LoadGeometry(path);
    const auto cells = finite_cell_benchmark::Resolution(geometry.Extent, resolution);
    const auto domain = modal::MakeTriangleSurfaceDomain(geometry.Boundary.Points, geometry.Boundary.Triangles);
    const auto operation = modal::BuildFiniteCellOperator(
        domain, Material,
        {.Cells = cells, .CutDepth = 2, .FictitiousScale = 1e-8, .PaddingCells = 0.25}
    );
    if (operation.Dofs() <= SolvedModeCount + 4) throw std::runtime_error("grid has insufficient degrees of freedom");
    const auto assembled = modal::finite_cell::SolveAssembledCholesky(operation, SolvedModeCount, Shift, 1e-9, 1000);
    const auto production = modal::SolveFiniteCellEigenpairs(operation, SolvedModeCount, Shift, 1e-8, max_iterations);
    if (assembled.Eigenvalues.size() != SolvedModeCount || production.Eigenvalues.size() != SolvedModeCount)
        throw std::runtime_error("assembled or production solve did not converge");
    const auto assembled_certification = modal::CertifyFiniteCellEigenpairs(operation, assembled.Eigenvalues, assembled.Eigenvectors);
    const auto production_certification = modal::CertifyFiniteCellEigenpairs(operation, production.Eigenvalues, production.Eigenvectors);
    const double spectrum = (production.Eigenvalues.segment(6, AcceptedModeCount - 6) - assembled.Eigenvalues.segment(6, AcceptedModeCount - 6)).norm() /
        assembled.Eigenvalues.segment(6, AcceptedModeCount - 6).norm();
    const double residual = production_certification.RelativeResiduals.segment(6, AcceptedModeCount - 6).maxCoeff();
    const double assembled_residual = assembled_certification.RelativeResiduals.segment(6, AcceptedModeCount - 6).maxCoeff();
    const auto shapes = finite_cell_benchmark::CompareSameDiscretizationModeShapes(
        operation, assembled.Eigenvalues, assembled.Eigenvectors, production.Eigenvectors, 6, AcceptedModeCount
    );
    const auto fractions = PhysicalCellFractions(operation);
    std::println(
        "real-mesh object={} triangles={} grid={}x{}x{} dofs={} active={} interior={} cut={} fill={:.3f} cell_fraction={:.2e}/{:.3f} slivers={} iterations={} failed_factor_free={}/{}/{:.3e} spectrum={:.3e} residual={:.3e} orthogonality={:.3e} assembled={:.3e}/{:.3e} cluster_mac={:.6f}",
        path.parent_path().filename().string() + "/" + path.stem().string(), geometry.Boundary.Triangles.size() / 3,
        cells.x, cells.y, cells.z, operation.Dofs(), operation.Profile.ActiveCells,
        operation.Profile.ActiveCells - operation.Profile.CutCells, operation.Profile.CutCells,
        operation.Profile.PhysicalVolume / (geometry.Extent.x * geometry.Extent.y * geometry.Extent.z), fractions.Minimum,
        fractions.Median, fractions.BelowOnePercent, production.Iterations,
        production.Profile.FailedFactorFreeIterations, production.Profile.FailedFactorFreeStagnated,
        production.Profile.FailedFactorFreeResidual, spectrum, residual,
        production_certification.MassOrthogonalityError, assembled_residual,
        assembled_certification.MassOrthogonalityError, shapes.ClusterMacMinimum
    );
    return {
        .Passed = spectrum < 1e-9 && residual < 1e-8 && production_certification.MassOrthogonalityError < 1e-9 &&
            assembled_residual < 1e-8 && assembled_certification.MassOrthogonalityError < 1e-9 &&
            shapes.ClusterMacMinimum > 0.99999,
        .FellBack = production.Profile.FailedFactorFreeIterations > 0,
        .Stagnated = production.Profile.FailedFactorFreeStagnated,
    };
}
} // namespace

int main(int argc, char **argv) try {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    fs::path root{FASTFEM_TET_CORPUS_DIR};
    uint32_t resolution{8}, max_iterations{100};
    std::optional<uint32_t> expected_fallbacks, expected_stagnations;
    std::vector<std::string> datasets;
    std::set<std::string> names;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string_view option{argv[argument]};
        if (option == "--corpus-dir" && argument + 1 < argc) root = argv[++argument];
        else if (option == "--resolution" && argument + 1 < argc) resolution = Parse(argv[++argument], "resolution");
        else if (option == "--max-iterations" && argument + 1 < argc) max_iterations = Parse(argv[++argument], "maximum iterations");
        else if (option == "--expected-fallbacks" && argument + 1 < argc) expected_fallbacks = Parse(argv[++argument], "expected fallback count");
        else if (option == "--expected-stagnations" && argument + 1 < argc) expected_stagnations = Parse(argv[++argument], "expected stagnation count");
        else if (option == "--dataset" && argument + 1 < argc) datasets.emplace_back(argv[++argument]);
        else if (option == "--help") {
            std::println("usage: {} [--corpus-dir PATH] [--resolution N] [--max-iterations N] [--expected-fallbacks N] [--expected-stagnations N] [--dataset NAME] [object ...]", argv[0]);
            return 0;
        } else names.emplace(option);
    }
    if (!fs::is_directory(root)) {
        std::println("finite-cell real-mesh corpus is absent: {} (run script/SetupTetCorpus)", root.string());
        return 77;
    }
    const auto objects = Objects(root, datasets, names);
    if (objects.empty()) throw std::invalid_argument("no matching OBJ files in " + root.string());
    std::println("finite-cell real-mesh consolidation: {} objects at longitudinal resolution {}", objects.size(), resolution);
    uint32_t failures{}, fallbacks{}, stagnations{};
    for (const auto &object : objects)
        try {
            const auto result = Run(object, resolution, max_iterations);
            failures += !result.Passed;
            fallbacks += result.FellBack;
            stagnations += result.Stagnated;
        } catch (const std::exception &error) {
            std::println(stderr, "real-mesh object={} failed: {}", object.string(), error.what());
            ++failures;
        }
    if (expected_fallbacks && fallbacks != *expected_fallbacks) {
        std::println(stderr, "finite-cell real-mesh consolidation: expected {} fallbacks, got {}", *expected_fallbacks, fallbacks);
        ++failures;
    }
    if (expected_stagnations && stagnations != *expected_stagnations) {
        std::println(stderr, "finite-cell real-mesh consolidation: expected {} stagnations, got {}", *expected_stagnations, stagnations);
        ++failures;
    }
    std::println(
        "finite-cell real-mesh consolidation: {} objects, {} fallbacks, {} stagnations, {} failures",
        objects.size(), fallbacks, stagnations, failures
    );
    return failures ? 1 : 0;
} catch (const std::exception &error) {
    std::println(stderr, "{}", error.what());
    return 2;
}
