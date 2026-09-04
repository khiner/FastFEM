#include "audio/FiniteCell.h"
#include "FiniteCellBenchmarkGeometry.h"
#include "ModeShapeComparison.h"
#include "RunSuites.h"
#include "TetReference.h"
#include "audio/FiniteCellBlockEigensolver.h"
#include "audio/finite_cell/AssembledCholesky.h"
#include "audio/finite_cell/OctreeQuadrature.h"
#include "numeric/Accelerate.h"

#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <print>
#include <vector>

using namespace boost::ut;

namespace {
constexpr AcousticMaterialProperties Material{.Density = 1000, .YoungModulus = 1e7, .PoissonRatio = 0, .Alpha = 0, .Beta = 0};

finite_cell_benchmark::Surface GriddedBox(dvec3 extent, uvec3 cells) {
    finite_cell_benchmark::Surface surface;
    std::map<std::array<uint32_t, 3>, uint32_t> nodes;
    const auto Node = [&](uvec3 coordinate) {
        const std::array key{coordinate.x, coordinate.y, coordinate.z};
        const auto [entry, inserted] = nodes.try_emplace(key, uint32_t(surface.Points.size()));
        if (inserted) surface.Points.push_back(extent * dvec3{coordinate} / dvec3{cells});
        return entry->second;
    };
    const auto Face = [&](uint32_t u_count, uint32_t v_count, auto point) {
        for (uint32_t u = 0; u < u_count; ++u)
            for (uint32_t v = 0; v < v_count; ++v) {
                const uint32_t a = Node(point(u, v)), b = Node(point(u + 1, v));
                const uint32_t c = Node(point(u + 1, v + 1)), d = Node(point(u, v + 1));
                surface.Triangles.insert(surface.Triangles.end(), {a, b, c, a, c, d});
            }
    };
    Face(cells.x, cells.y, [&](uint32_t x, uint32_t y) { return uvec3{x, y, 0}; });
    Face(cells.x, cells.y, [&](uint32_t x, uint32_t y) { return uvec3{x, y, cells.z}; });
    Face(cells.x, cells.z, [&](uint32_t x, uint32_t z) { return uvec3{x, 0, z}; });
    Face(cells.x, cells.z, [&](uint32_t x, uint32_t z) { return uvec3{x, cells.y, z}; });
    Face(cells.y, cells.z, [&](uint32_t y, uint32_t z) { return uvec3{0, y, z}; });
    Face(cells.y, cells.z, [&](uint32_t y, uint32_t z) { return uvec3{cells.x, y, z}; });
    return surface;
}

enum struct BarFamily : uint32_t { Longitudinal,
                                   Torsional,
                                   Bending,
                                   Count };

struct BarFrequencies {
    std::array<double, uint32_t(BarFamily::Count)> Values{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };

    double &operator[](BarFamily family) { return Values[uint32_t(family)]; }
    double operator[](BarFamily family) const { return Values[uint32_t(family)]; }
};

BarFrequencies ClassifyBarFrequencies(
    const Eigen::MatrixXd &sampled, const Eigen::VectorXd &eigenvalues, dvec3 extent, uvec3 samples,
    uint32_t first_mode = 6
) {
    BarFrequencies result;
    for (Eigen::Index mode = first_mode; mode < eigenvalues.size(); ++mode) {
        const Eigen::Index column = mode - first_mode;
        double axial{}, lateral_y{}, lateral_z{}, rotation{}, total{};
        for (uint32_t x = 0; x < samples.x; ++x) {
            double circulation{}, radius_squared{};
            for (uint32_t y = 0; y < samples.y; ++y)
                for (uint32_t z = 0; z < samples.z; ++z) {
                    const uint32_t sample = (x * samples.y + y) * samples.z + z;
                    const dvec3 displacement{
                        sampled(3 * sample, column), sampled(3 * sample + 1, column), sampled(3 * sample + 2, column)
                    };
                    const double ry = extent.y * (double(y) + 0.5) / samples.y - 0.5 * extent.y;
                    const double rz = extent.z * (double(z) + 0.5) / samples.z - 0.5 * extent.z;
                    axial += displacement.x * displacement.x;
                    lateral_y += displacement.y * displacement.y;
                    lateral_z += displacement.z * displacement.z;
                    total += numeric::Dot(displacement, displacement);
                    circulation += ry * displacement.z - rz * displacement.y;
                    radius_squared += ry * ry + rz * rz;
                }
            if (radius_squared > 0) rotation += circulation * circulation / radius_squared;
        }
        if (total <= 0) continue;
        const double frequency = std::sqrt(std::max(0.0, eigenvalues[mode])) / (2 * std::numbers::pi);
        if (axial / total > 0.8)
            result[BarFamily::Longitudinal] = std::min(result[BarFamily::Longitudinal], frequency);
        else if (rotation / total > 0.75)
            result[BarFamily::Torsional] = std::min(result[BarFamily::Torsional], frequency);
        else if ((lateral_y + lateral_z) / total > 0.6 && rotation / total < 0.5)
            result[BarFamily::Bending] = std::min(result[BarFamily::Bending], frequency);
    }
    return result;
}

std::pair<BarFrequencies, BarFrequencies> SampleBarFrequencies(
    const modal::FiniteCellOperator &finite, const modal::FiniteCellBlockResult &finite_modes,
    const TetMesh &mesh, const modal::Tet10Assembler &tet, const TetReferenceEigenpairs &tet_modes, dvec3 extent
) {
    constexpr uvec3 samples{9, 3, 3};
    std::vector<finite_cell_benchmark::InterpolationStencil> finite_stencils, tet_stencils;
    finite_stencils.reserve(samples.x * samples.y * samples.z);
    tet_stencils.reserve(samples.x * samples.y * samples.z);
    for (uint32_t x = 0; x < samples.x; ++x)
        for (uint32_t y = 0; y < samples.y; ++y)
            for (uint32_t z = 0; z < samples.z; ++z) {
                const dvec3 point = extent * (dvec3{double(x), double(y), double(z)} + 0.5) / dvec3{samples};
                const auto finite_stencil = finite_cell_benchmark::FiniteCellStencil(finite, point);
                const auto tet_stencil = finite_cell_benchmark::Tet10Stencil(mesh, tet, point);
                expect(bool(finite_stencil));
                expect(bool(tet_stencil));
                if (!finite_stencil || !tet_stencil) continue;
                finite_stencils.push_back(*finite_stencil);
                tet_stencils.push_back(*tet_stencil);
            }
    const Eigen::MatrixXd finite_sampled = finite_cell_benchmark::SampleModes(finite_stencils, finite_modes.Eigenvectors, 6);
    const Eigen::MatrixXd tet_sampled = finite_cell_benchmark::SampleModes(tet_stencils, tet_modes.Eigenvectors, 6);
    return {
        ClassifyBarFrequencies(finite_sampled, finite_modes.Eigenvalues, extent, samples),
        ClassifyBarFrequencies(tet_sampled, tet_modes.Eigenvalues, extent, samples),
    };
}
} // namespace

suite FiniteCellTests = [] {
    "Accelerate symmetric cross-Gram matches explicit symmetrization"_test = [] {
        Eigen::MatrixXd a(257, 37), b(257, 37), accelerated(37, 37);
        for (Eigen::Index column = 0; column < a.cols(); ++column)
            for (Eigen::Index row = 0; row < a.rows(); ++row) {
                a(row, column) = std::sin(0.17 * double(row + 1) + 0.31 * double(column + 1));
                b(row, column) = std::cos(0.23 * double(row + 1) - 0.29 * double(column + 1));
            }
        numeric::SymmetricCrossGram(a.data(), b.data(), accelerated.data(), uint32_t(a.rows()), uint32_t(a.cols()));
        const Eigen::MatrixXd product = a.transpose() * b;
        const Eigen::MatrixXd expected = 0.5 * (product + product.transpose());
        expect(lt((accelerated - expected).norm() / expected.norm(), 2e-15));
    };

    "cross-discretization mode-shape interpolation is affine exact"_test = [] {
        const auto finite = modal::BuildFiniteCellOperator(
            modal::MakeBoxDomain({}, {0.3, 0.05, 0.02}), Material,
            {.Cells = {3, 2, 2}, .CutDepth = 1, .FictitiousScale = 1e-8, .PaddingCells = 0}
        );
        const TetMesh mesh = MakeStructuredBar(3, 2, 2);
        const modal::Tet10Assembler tet{mesh, Material};
        const dvec3 point{0.137, 0.023, 0.011};
        const auto finite_stencil = finite_cell_benchmark::FiniteCellStencil(finite, point);
        const auto tet_stencil = finite_cell_benchmark::Tet10Stencil(mesh, tet, point);
        expect(bool(finite_stencil));
        expect(bool(tet_stencil));
        const auto Interpolate = [&](const auto &stencil, const std::vector<dvec3> &nodes) {
            dvec3 result{};
            for (uint32_t node = 0; node < stencil.Count; ++node) result += stencil.Weights[node] * nodes[stencil.Nodes[node]];
            return result;
        };
        expect(numeric::Length(Interpolate(*finite_stencil, finite.Nodes) - point) < 1e-14);
        constexpr uint32_t edge_corners[6][2]{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
        std::vector<dvec3> tet_nodes(tet.NumNodes);
        std::copy(mesh.Points.begin(), mesh.Points.end(), tet_nodes.begin());
        for (uint32_t element = 0; element < tet.Elements().size(); ++element)
            for (uint32_t edge = 0; edge < 6; ++edge) {
                const auto &corners = mesh.Tets[element];
                tet_nodes[tet.Elements()[element].Nodes[4 + edge]] =
                    0.5 * (mesh.Points[corners[edge_corners[edge][0]]] + mesh.Points[corners[edge_corners[edge][1]]]);
            }
        expect(numeric::Length(Interpolate(*tet_stencil, tet_nodes) - point) < 1e-14);
    };

    "exact implicit domains provide signed distances and tight bounds"_test = [] {
        const auto sphere = modal::MakeSphereDomain({1, 2, 3}, 2);
        expect(numeric::Length(sphere.Min - dvec3{-1, 0, 1}) < 1e-14);
        expect(numeric::Length(sphere.Max - dvec3{3, 4, 5}) < 1e-14);
        expect(std::abs(sphere.SignedDistance({1, 2, 3}) + 2) < 1e-14);
        expect(std::abs(sphere.SignedDistance({3, 2, 3})) < 1e-14);
        expect(std::abs(sphere.SignedDistance({4, 2, 3}) - 1) < 1e-14);
        expect(sphere.ClassifyBox({1, 2, 3}, dvec3{0.5}) == modal::DomainRegion::Inside);
        expect(sphere.ClassifyBox({4, 2, 3}, dvec3{0.5}) == modal::DomainRegion::Outside);
        expect(sphere.ClassifyBox({3, 2, 3}, dvec3{0.5}) == modal::DomainRegion::Cut);

        const auto shell = modal::MakeSphericalShellDomain({}, 1, 3);
        expect(std::abs(shell.SignedDistance({0, 0, 0}) - 1) < 1e-14);
        expect(std::abs(shell.SignedDistance({1, 0, 0})) < 1e-14);
        expect(std::abs(shell.SignedDistance({2, 0, 0}) + 1) < 1e-14);
        expect(std::abs(shell.SignedDistance({3, 0, 0})) < 1e-14);
        expect(shell.ClassifyBox({2, 0, 0}, dvec3{0.25}) == modal::DomainRegion::Inside);
        expect(shell.ClassifyBox({}, dvec3{0.25}) == modal::DomainRegion::Outside);
        expect(shell.ClassifyBox({1, 0, 0}, dvec3{0.25}) == modal::DomainRegion::Cut);

        const auto cylinder = modal::MakeCylinderDomain({}, 2, 6);
        expect(numeric::Length(cylinder.Min - dvec3{-2, -2, -3}) < 1e-14);
        expect(numeric::Length(cylinder.Max - dvec3{2, 2, 3}) < 1e-14);
        expect(std::abs(cylinder.SignedDistance({0, 0, 0}) + 2) < 1e-14);
        expect(std::abs(cylinder.SignedDistance({2, 0, 0})) < 1e-14);
        expect(std::abs(cylinder.SignedDistance({0, 0, 3})) < 1e-14);
        expect(std::abs(cylinder.SignedDistance({3, 0, 4}) - std::sqrt(2.0)) < 1e-14);
        expect(cylinder.ClassifyBox({}, dvec3{0.5}) == modal::DomainRegion::Inside);
        expect(cylinder.ClassifyBox({3, 0, 0}, dvec3{0.5}) == modal::DomainRegion::Outside);
        expect(cylinder.ClassifyBox({2, 0, 0}, dvec3{0.5}) == modal::DomainRegion::Cut);
    };

    "finite-cell benchmark geometries build mesh-faithful references"_test = [] {
        for (const std::string_view name : finite_cell_benchmark::GeometryNames) {
            const auto geometry = finite_cell_benchmark::MakeGeometry(name);
            const auto domain = modal::MakeTriangleSurfaceDomain(geometry.Boundary.Points, geometry.Boundary.Triangles);
            expect(domain.SignedDistance(geometry.InteriorPoint) < 0) << name;
            expect(domain.SignedDistance(domain.Max + numeric::Max(domain.Max - domain.Min, dvec3{1})) > 0) << name;

            const uvec3 resolution = finite_cell_benchmark::ReferenceResolution(geometry, 3);
            const auto reference = geometry.Reference(resolution.x, resolution.y, resolution.z);
            expect(!reference.Points.empty()) << name;
            expect(!reference.Tets.empty()) << name;
            double reference_volume{};
            for (const auto &tet : reference.Tets)
                for (const uint32_t point : tet) expect(point < reference.Points.size()) << name;
            for (const auto &tet : reference.Tets) {
                const dvec3 a = reference.Points[tet[0]] - reference.Points[tet[3]];
                const dvec3 b = reference.Points[tet[1]] - reference.Points[tet[3]];
                const dvec3 c = reference.Points[tet[2]] - reference.Points[tet[3]];
                reference_volume += std::abs(numeric::Dot(a, numeric::Cross(b, c))) / 6;
            }
            expect(std::abs(reference_volume / geometry.PhysicalVolume - 1) < 1e-10) << name;
        }
    };

    "Tet10 and finite cells track analytical bar modes on identical boundary meshes"_test = [] {
        constexpr dvec3 extent{0.3, 0.05, 0.05};
        constexpr uint32_t Count{20};
        constexpr std::array levels{uvec3{8, 2, 2}, uvec3{14, 3, 3}, uvec3{20, 4, 4}};
        constexpr std::array names{"longitudinal", "torsional", "bending"};
        constexpr double torsion_over_polar = 0.140577 * 6;
        const BarFrequencies theory{{
            std::sqrt(Material.YoungModulus / Material.Density) / (2 * extent.x),
            std::sqrt(Material.Mu() / Material.Density * torsion_over_polar) / (2 * extent.x),
            4.73004074 * 4.73004074 / (2 * std::numbers::pi) *
                std::sqrt(Material.YoungModulus / Material.Density) * extent.z /
                (std::sqrt(12.0) * extent.x * extent.x),
        }};
        std::array<BarFrequencies, levels.size()> finite_frequencies_by_level, tet_frequencies_by_level;
        std::array<BarFrequencies, levels.size()> finite_errors, tet_errors;
        const double shift = std::pow(2 * std::numbers::pi * 20, 2);
        for (uint32_t level = 0; level < levels.size(); ++level) {
            const uvec3 cells = levels[level];
            const auto surface = GriddedBox(extent, cells);
            const auto domain = modal::MakeTriangleSurfaceDomain(surface.Points, surface.Triangles);
            const TetMesh mesh = finite_cell_benchmark::TetrahedralReference(surface, cells.x, cells.y, cells.z);
            const modal::Tet10Assembler tet{mesh, Material};
            const auto finite = modal::BuildFiniteCellOperator(
                domain, Material,
                {.Cells = cells, .CutDepth = 3, .FictitiousScale = 1e-8, .PaddingCells = 0.25}
            );
            const auto tet_modes = SolveTetReference(mesh, Material, Count, shift, 1e-8, 150);
            const auto finite_modes = modal::SolveFiniteCellBlock(finite, Count, shift, 1e-8, 150);
            expect(tet_modes.Eigenvalues.size() == Count) << level;
            expect(finite_modes.Eigenvalues.size() == Count) << level;
            if (tet_modes.Eigenvalues.size() != Count || finite_modes.Eigenvalues.size() != Count) continue;
            const auto finite_certification = modal::CertifyFiniteCellEigenpairs(finite, finite_modes.Eigenvalues, finite_modes.Eigenvectors);
            expect(tet_modes.RelativeResidual < 1e-7) << level;
            expect(finite_certification.RelativeResiduals.tail(Count - 6).maxCoeff() < 1e-7) << level;
            const auto [finite_frequencies, tet_frequencies] =
                SampleBarFrequencies(finite, finite_modes, mesh, tet, tet_modes, extent);
            finite_frequencies_by_level[level] = finite_frequencies;
            tet_frequencies_by_level[level] = tet_frequencies;
            std::println(
                "analytical bar {}x{}x{}: surface={} triangles={} tet-dofs={} finite-dofs={}",
                cells.x, cells.y, cells.z, surface.Points.size(), surface.Triangles.size() / 3, tet.Dofs(), finite.Dofs()
            );
            for (uint32_t family = 0; family < uint32_t(BarFamily::Count); ++family) {
                const BarFamily selected = BarFamily(family);
                const double finite_frequency = finite_frequencies[selected], tet_frequency = tet_frequencies[selected];
                expect(std::isfinite(finite_frequency)) << names[family] << level;
                expect(std::isfinite(tet_frequency)) << names[family] << level;
                finite_errors[level][selected] = std::abs(finite_frequency / theory[selected] - 1);
                tet_errors[level][selected] = std::abs(tet_frequency / theory[selected] - 1);
                std::println(
                    "  {:>12}: theory={:9.3f} Hz tet10={:9.3f} Hz abs={:8.3f} Hz rel={:7.3f}% "
                    "finite={:9.3f} Hz abs={:8.3f} Hz rel={:7.3f}% cross={:7.3f}%",
                    names[family], theory[selected], tet_frequency, std::abs(tet_frequency - theory[selected]),
                    100 * tet_errors[level][selected], finite_frequency, std::abs(finite_frequency - theory[selected]),
                    100 * finite_errors[level][selected], 100 * std::abs(finite_frequency / tet_frequency - 1)
                );
            }
        }
        constexpr BarFrequencies tolerances{{0.03, 0.08, 0.10}};
        for (uint32_t family = 0; family < uint32_t(BarFamily::Count); ++family) {
            const BarFamily selected = BarFamily(family);
            expect(tet_errors.back()[selected] < tolerances[selected]) << names[family];
            expect(finite_errors.back()[selected] < tolerances[selected]) << names[family];
            expect(
                std::abs(tet_frequencies_by_level[2][selected] - tet_frequencies_by_level[1][selected]) <
                std::abs(tet_frequencies_by_level[1][selected] - tet_frequencies_by_level[0][selected])
            ) << names[family];
            expect(finite_errors.back()[selected] < finite_errors.front()[selected]) << names[family];
        }
    };

    "finite-cell block solve matches assembled FP64 across audio geometries"_test = [] {
        constexpr uint32_t Count{18};
        const double shift = std::pow(2 * std::numbers::pi * 20, 2);
        for (const std::string_view name : finite_cell_benchmark::AudioGeometryNames) {
            const auto geometry = finite_cell_benchmark::MakeGeometry(name);
            const auto domain = modal::MakeTriangleSurfaceDomain(geometry.Boundary.Points, geometry.Boundary.Triangles);
            const auto operation = modal::BuildFiniteCellOperator(
                domain, Material,
                {.Cells = finite_cell_benchmark::GridResolution(geometry, 6), .CutDepth = 2, .FictitiousScale = 1e-8, .PaddingCells = 0.25}
            );
            const auto assembled = modal::finite_cell::SolveAssembledCholesky(operation, Count, shift, 5e-9, 150);
            const auto preferred = modal::SolveFiniteCellBlock(operation, Count, shift, 5e-9, 150);
            expect(assembled.Eigenvalues.size() == Count) << name;
            expect(preferred.Eigenvalues.size() == Count) << name;
            if (assembled.Eigenvalues.size() != Count || preferred.Eigenvalues.size() != Count) continue;
            const auto preferred_certification = modal::CertifyFiniteCellEigenpairs(operation, preferred.Eigenvalues, preferred.Eigenvectors);
            const double spectrum_error = (preferred.Eigenvalues.tail(Count - 6) - assembled.Eigenvalues.tail(Count - 6)).norm() /
                assembled.Eigenvalues.tail(Count - 6).norm();
            const double residual = preferred_certification.RelativeResiduals.tail(Count - 6).maxCoeff();
            const auto shapes = finite_cell_benchmark::CompareSameDiscretizationModeShapes(
                operation, assembled.Eigenvalues, assembled.Eigenvectors, preferred.Eigenvectors
            );
            std::println(
                "audio geometry {}: dofs={} cut={} iterations={} spectrum={:.3e} residual={:.3e} orthogonality={:.3e} paired_mac={:.6f} cluster_mac={:.6f}",
                name, operation.Dofs(), operation.Profile.CutCells, preferred.Iterations, spectrum_error, residual,
                preferred_certification.MassOrthogonalityError, shapes.PairedMacMinimum, shapes.ClusterMacMinimum
            );
            expect(spectrum_error < 1e-10) << name;
            expect(residual < 1e-8) << name;
            expect(preferred_certification.MassOrthogonalityError < 1e-10) << name;
            expect(shapes.ClusterMacMinimum > 0.999999) << name;
        }
    };

    "matrix-free finite-cell actions match assembly"_test = [] {
        constexpr uint32_t Width{3};
        {
            const auto operation = modal::BuildFiniteCellOperator(
                modal::MakeBoxDomain({}, {0.3, 0.08, 0.05}), Material,
                {.Cells = {4, 3, 2}, .CutDepth = 2, .FictitiousScale = 1e-8, .PaddingCells = 0.27}
            );
            const auto assembled = operation.AssembleLower();
            Eigen::MatrixXd input(operation.Dofs(), Width), mass(operation.Dofs(), Width);
            Eigen::MatrixXd paired_mass(operation.Dofs(), Width), paired_shifted(operation.Dofs(), Width);
            Eigen::MatrixXd expanded_packed_cut_mass(operation.Dofs(), Width), expanded_packed_cut_shifted(operation.Dofs(), Width);
            Eigen::MatrixXd stiffness(operation.Dofs(), Width), shifted(operation.Dofs(), Width);
            for (Eigen::Index column = 0; column < input.cols(); ++column)
                for (Eigen::Index row = 0; row < input.rows(); ++row)
                    input(row, column) = std::sin(0.31 * double(row + 1) + 0.17 * double(column + 1));
            operation.ApplyMass(input.data(), mass.data(), Width);
            operation.ApplyStiffness(input.data(), stiffness.data(), Width);
            operation.ApplyShifted(input.data(), shifted.data(), Width, 19);
            operation.ApplyMassShifted(input.data(), paired_mass.data(), paired_shifted.data(), Width, 19);
            auto packed_cut = operation.BuildPackedCutOperators(19);
            operation.ApplyMassShiftedExpandedPackedCut(
                packed_cut, input.data(), expanded_packed_cut_mass.data(), expanded_packed_cut_shifted.data(), Width
            );
            const Eigen::MatrixXd assembled_mass = assembled.Mass.selfadjointView<Eigen::Lower>() * input;
            const Eigen::MatrixXd assembled_stiffness = assembled.Stiffness.selfadjointView<Eigen::Lower>() * input;
            const double mass_error = (mass - assembled_mass).norm() / assembled_mass.norm();
            const double stiffness_error = (stiffness - assembled_stiffness).norm() / assembled_stiffness.norm();
            const double shifted_error = (shifted - assembled_stiffness - 19 * assembled_mass).norm() /
                (assembled_stiffness + 19 * assembled_mass).norm();
            std::println(
                "Q2 finite-cell action error: mass {:.3e}, stiffness {:.3e}, shifted {:.3e}",
                mass_error, stiffness_error, shifted_error
            );
            expect(mass_error < 2e-14);
            expect(stiffness_error < 2e-14);
            expect(shifted_error < 2e-14);
            expect((paired_mass.array() == mass.array()).all());
            expect((paired_shifted.array() == shifted.array()).all());
            expect((expanded_packed_cut_mass - paired_mass).norm() / paired_mass.norm() < 2e-14);
            expect((expanded_packed_cut_shifted - paired_shifted).norm() / paired_shifted.norm() < 2e-14);
            for (uint32_t node = 0; node < operation.Nodes.size(); ++node) {
                uint32_t colors{};
                for (uint32_t entry = operation.NodeOccurrenceOffsets[node]; entry < operation.NodeOccurrenceOffsets[node + 1]; ++entry) {
                    const uint32_t color = operation.Cells[operation.NodeOccurrences[entry] >> 5].Color;
                    expect(!(colors & (1u << color)));
                    colors |= 1u << color;
                }
            }
        }
    };

    "signed moment fitting preserves linear modal operators"_test = [] {
        const auto domain = modal::MakeSphereDomain({}, 0.1);
        const modal::FiniteCellConfig config{
            .Cells = {4, 4, 4},
            .CutDepth = 3,
            .FictitiousScale = 1e-8,
            .PaddingCells = 0.2,
        };
        const auto octree = modal::finite_cell::BuildOctreeOperator(domain, Material, config);
        const auto fitted = modal::BuildFiniteCellOperator(domain, Material, config);
        const auto repeated = modal::BuildFiniteCellOperator(domain, Material, config);
        expect(repeated.Quadrature.size() == fitted.Quadrature.size());
        if (repeated.Quadrature.size() == fitted.Quadrature.size())
            for (uint32_t point = 0; point < fitted.Quadrature.size(); ++point) {
                expect(repeated.Quadrature[point].Reference == fitted.Quadrature[point].Reference);
                expect(repeated.Quadrature[point].Weight == fitted.Quadrature[point].Weight);
                expect(repeated.Quadrature[point].Fictitious == fitted.Quadrature[point].Fictitious);
            }
        constexpr uint32_t Width{3};
        Eigen::MatrixXd input(octree.Dofs(), Width), expected(octree.Dofs(), Width), actual(octree.Dofs(), Width);
        for (Eigen::Index column = 0; column < input.cols(); ++column)
            for (Eigen::Index row = 0; row < input.rows(); ++row)
                input(row, column) = std::cos(0.17 * double(row + 1) + 0.11 * double(column + 1));
        octree.ApplyShifted(input.data(), expected.data(), Width, 13);
        fitted.ApplyShifted(input.data(), actual.data(), Width, 13);
        const double action_error = (actual - expected).norm() / expected.norm();
        const uint64_t negative_weights = std::ranges::count_if(
            fitted.Quadrature, [](const auto &point) { return point.Weight < 0; }
        );

        const auto scaled_octree = octree.WithFictitiousScale(1e-4);
        const auto scaled_fitted = fitted.WithFictitiousScale(1e-4);
        scaled_octree.ApplyShifted(input.data(), expected.data(), Width, 13);
        scaled_fitted.ApplyShifted(input.data(), actual.data(), Width, 13);
        const double scaled_error = (actual - expected).norm() / expected.norm();

        constexpr uint32_t ModeCount{18}, ComparedCount{12};
        const double shift = std::pow(2 * std::numbers::pi * 20, 2);
        const auto octree_modes = modal::finite_cell::SolveAssembledCholesky(octree, ModeCount, shift, 1e-8, 100);
        const auto fitted_modes = modal::finite_cell::SolveAssembledCholesky(fitted, ModeCount, shift, 1e-8, 100);
        expect(octree_modes.Eigenvalues.size() == ModeCount);
        expect(fitted_modes.Eigenvalues.size() == ModeCount);
        double spectrum_error{std::numeric_limits<double>::infinity()};
        if (octree_modes.Eigenvalues.size() == ModeCount && fitted_modes.Eigenvalues.size() == ModeCount)
            spectrum_error =
                (fitted_modes.Eigenvalues.segment(6, ComparedCount - 6) -
                 octree_modes.Eigenvalues.segment(6, ComparedCount - 6))
                    .norm() /
                octree_modes.Eigenvalues.segment(6, ComparedCount - 6).norm();
        std::println(
            "signed moment fitting: {} / {} points, {} negative weights, build {:.3f} / {:.3f} s, "
            "action {:.3e}, scaled {:.3e}, spectrum {:.3e}",
            fitted.Profile.QuadraturePoints, octree.Profile.QuadraturePoints, negative_weights,
            fitted.Profile.Assemble, octree.Profile.Assemble, action_error, scaled_error, spectrum_error
        );
        expect(fitted.Profile.MomentFittedCells > 0u);
        expect(fitted.Profile.QuadraturePoints < octree.Profile.QuadraturePoints);
        expect(negative_weights > 0u);
        expect(action_error < 1e-9);
        expect(scaled_error < 1e-9);
        expect(spectrum_error < 1e-10);
        if (fitted_modes.Eigenvalues.size() == ModeCount) {
            const auto fitted_certification = modal::CertifyFiniteCellEigenpairs(fitted, fitted_modes.Eigenvalues, fitted_modes.Eigenvectors);
            expect(fitted_certification.RelativeResiduals.tail(ModeCount - 6).maxCoeff() < 1e-8);
            expect(fitted_certification.MassOrthogonalityError < 1e-10);
        }
    };

    "Cartesian P1 transfer is adjoint and Galerkin-consistent"_test = [] {
        constexpr uint32_t Width{3};
        constexpr double Alpha{19};
        {
            const auto operation = modal::BuildFiniteCellOperator(
                modal::MakeBoxDomain({}, {0.3, 0.08, 0.05}), Material,
                {.Cells = {4, 3, 2}, .CutDepth = 2, .FictitiousScale = 1e-8, .PaddingCells = 0.27}
            );
            const auto stabilized = operation.WithFictitiousScale(1e-4);
            expect(stabilized.FictitiousScale == 1e-4);
            double scale_error{};
            for (uint32_t point = 0; point < operation.Quadrature.size(); ++point) {
                const double ratio = stabilized.Quadrature[point].Weight / operation.Quadrature[point].Weight;
                scale_error = std::max(scale_error, std::abs(ratio - (operation.Quadrature[point].Fictitious ? 1e4 : 1)));
            }
            expect(scale_error < 1e-10);
            const uint32_t coarse_dofs = 3 * operation.NumP1Nodes;
            Eigen::MatrixXd coarse(coarse_dofs, Width), fine(operation.Dofs(), Width);
            for (Eigen::Index column = 0; column < Width; ++column) {
                for (Eigen::Index row = 0; row < coarse.rows(); ++row)
                    coarse(row, column) = std::sin(0.19 * double(row + 1) + 0.11 * double(column + 1));
                for (Eigen::Index row = 0; row < fine.rows(); ++row)
                    fine(row, column) = std::cos(0.23 * double(row + 1) - 0.07 * double(column + 1));
            }
            Eigen::MatrixXd prolonged(operation.Dofs(), Width), restricted(coarse_dofs, Width);
            operation.ProlongP1(coarse.data(), prolonged.data(), Width);
            operation.RestrictP1(fine.data(), restricted.data(), Width);
            const double adjoint_error = std::abs(prolonged.cwiseProduct(fine).sum() - coarse.cwiseProduct(restricted).sum()) /
                std::max(std::abs(prolonged.cwiseProduct(fine).sum()), 1.0);

            Eigen::MatrixXd fine_action(operation.Dofs(), Width), coarse_action(coarse_dofs, Width);
            operation.ApplyShifted(prolonged.data(), fine_action.data(), Width, Alpha);
            operation.RestrictP1(fine_action.data(), coarse_action.data(), Width);
            const auto coarse_matrix = operation.AssembleP1ShiftedLower(Alpha);
            const Eigen::MatrixXd assembled_action = coarse_matrix.selfadjointView<Eigen::Lower>() * coarse;
            const double galerkin_error = (coarse_action - assembled_action).norm() / assembled_action.norm();

            const auto assembled = operation.AssembleLower();
            const Eigen::VectorXd diagonal = operation.ShiftedDiagonal(Alpha);
            const Eigen::VectorXd assembled_diagonal = assembled.Stiffness.diagonal() + Alpha * assembled.Mass.diagonal();
            const double diagonal_error = (diagonal - assembled_diagonal).norm() / assembled_diagonal.norm();
            std::println(
                "Q2 P1 transfer: {} fine / {} coarse dofs, adjoint {:.3e}, Galerkin {:.3e}, diagonal {:.3e}",
                operation.Dofs(), coarse_dofs, adjoint_error, galerkin_error, diagonal_error
            );
            expect(adjoint_error < 2e-14);
            expect(galerkin_error < 2e-14);
            expect(diagonal_error < 2e-14);
        }
    };

    "Q2 finite cells preserve rigid modes and exact full-cell volume"_test = [] {
        const dvec3 extent{0.3, 0.05, 0.05};
        const modal::ImplicitDomain full_grid{{}, extent, [](const dvec3 &) { return -1.0; }};
        const auto operation = modal::BuildFiniteCellOperator(
            full_grid, Material,
            {.Cells = {4, 2, 2}, .CutDepth = 1, .FictitiousScale = 1e-8, .PaddingCells = 0}
        );
        const auto result = modal::SolveFiniteCellBlock(operation, 12, std::pow(2 * std::numbers::pi * 20, 2), 1e-8, 150);
        expect(result.Eigenvalues.size() == 12_i);
        if (result.Eigenvalues.size() != 12) return;
        expect(operation.Profile.CutCells == 0_u);
        expect(std::abs(operation.Profile.PhysicalVolume / (extent.x * extent.y * extent.z) - 1) < 1e-12);
        expect(result.Eigenvalues.head(6).cwiseAbs().maxCoeff() < 1e-3);
        const auto certification = modal::CertifyFiniteCellEigenpairs(operation, result.Eigenvalues, result.Eigenvectors);
        expect(certification.RelativeResiduals.tail(6).maxCoeff() < 1e-8);
        expect(certification.MassOrthogonalityError < 1e-8);
        const double residual_difference = (certification.RelativeResiduals.tail(6) - result.RelativeResiduals.tail(6)).norm();
        expect(residual_difference < 1e-12) << residual_difference;
    };

    "finite cells recover free bar volume and longitudinal frequency"_test = [] {
        const dvec3 extent{0.3, 0.05, 0.05};
        const auto operation = modal::BuildFiniteCellOperator(
            modal::MakeBoxDomain({}, extent), Material,
            {.Cells = {16, 4, 4}, .CutDepth = 4, .FictitiousScale = 1e-8, .PaddingCells = 0.31}
        );
        const auto result = modal::SolveFiniteCellBlock(operation, 20, std::pow(2 * std::numbers::pi * 20, 2), 1e-8, 150);
        expect(result.Eigenvalues.size() == 20_i);
        if (result.Eigenvalues.size() != 20) return;
        const double exact_volume = extent.x * extent.y * extent.z;
        const double volume_error = std::abs(operation.Profile.PhysicalVolume / exact_volume - 1);
        expect(volume_error < 0.015);
        expect(operation.Profile.CutCells > 0_u);
        const auto result_certification = modal::CertifyFiniteCellEigenpairs(operation, result.Eigenvalues, result.Eigenvectors);
        expect(result_certification.RelativeResiduals.tail(14).maxCoeff() < 1e-7);
        expect(result_certification.MassOrthogonalityError < 1e-8);

        double longitudinal = std::numeric_limits<double>::infinity();
        for (Eigen::Index mode = 6; mode < result.Eigenvalues.size(); ++mode) {
            const auto vector = result.Eigenvectors.col(mode);
            double axial{}, total{};
            for (Eigen::Index node = 0; node < vector.size() / 3; ++node) {
                axial += vector[3 * node] * vector[3 * node];
                total += vector.segment<3>(3 * node).squaredNorm();
            }
            if (axial / total > 0.8)
                longitudinal = std::min(longitudinal, std::sqrt(std::max(0.0, result.Eigenvalues[mode])) / (2 * std::numbers::pi));
        }
        const double exact = std::sqrt(Material.YoungModulus / Material.Density) / (2 * extent.x);
        const double frequency_error = std::abs(longitudinal / exact - 1);
        std::println("finite-cell bar: {} dofs, {} cut cells, volume error {:.3e}, longitudinal error {:.3e}", operation.Profile.Dofs, operation.Profile.CutCells, volume_error, frequency_error);
        expect(std::isfinite(longitudinal));
        expect(frequency_error < 0.06);
    };

    "triangle surface drives cut-cell integration without tetrahedra"_test = [] {
        const dvec3 extent{0.3, 0.08, 0.05};
        const auto surface = finite_cell_benchmark::BoxSurface(extent, 0.29, -0.17);
        const auto domain = modal::MakeTriangleSurfaceDomain(surface.Points, surface.Triangles);
        expect(domain.SignedDistance(0.5 * extent) < 0);
        expect(domain.SignedDistance({1, 1, 1}) > 0);
        expect(domain.ClassifyBox(0.5 * extent, dvec3{1e-3}) == modal::DomainRegion::Inside);
        expect(domain.ClassifyBox({1, 1, 1}, dvec3{1e-3}) == modal::DomainRegion::Outside);
        const auto operation = modal::BuildFiniteCellOperator(
            domain, Material, {.Cells = {12, 6, 5}, .CutDepth = 2, .FictitiousScale = 1e-8, .PaddingCells = 0.25}
        );
        auto conservative_domain = domain;
        conservative_domain.ClassifyBox = {};
        const auto conservative = modal::BuildFiniteCellOperator(
            conservative_domain, Material, {.Cells = {12, 6, 5}, .CutDepth = 2, .FictitiousScale = 1e-8, .PaddingCells = 0.25}
        );
        const auto assembled = operation.AssembleLower();
        const double exact_volume = extent.x * extent.y * extent.z;
        const double volume_error = std::abs(operation.Profile.PhysicalVolume / exact_volume - 1);
        std::println(
            "finite-cell surface: {} background / {} active / {} cut cells, {} / {} exact/conservative quadrature points, volume error {:.3e}",
            operation.Profile.BackgroundCells, operation.Profile.ActiveCells, operation.Profile.CutCells, operation.Profile.QuadraturePoints,
            conservative.Profile.QuadraturePoints, volume_error
        );
        expect(operation.Profile.ActiveCells < operation.Profile.BackgroundCells);
        expect(operation.Profile.CutCells > 0_u);
        expect(operation.Profile.Dofs > 0_u);
        expect(operation.Profile.ActiveCells <= conservative.Profile.ActiveCells);
        expect(operation.Profile.QuadraturePoints < conservative.Profile.QuadraturePoints);
        expect(std::abs(operation.Profile.PhysicalVolume - conservative.Profile.PhysicalVolume) < 1e-14);
        expect(volume_error < 0.04);
        expect((assembled.Mass.diagonal().array() > 0).all());
    };
};

int main() { return RunSuites(); }
