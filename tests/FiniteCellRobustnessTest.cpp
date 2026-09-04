#include "FiniteCellBenchmarkGeometry.h"
#include "ModeShapeComparison.h"
#include "RunSuites.h"
#include "audio/FiniteCell.h"
#include "audio/FiniteCellEigensolver.h"
#include "audio/finite_cell/AccelerateShiftInvert.h"
#include "audio/finite_cell/AssembledEigensolver.h"
#include "audio/finite_cell/EigenpairCertification.h"

#include <boost/ut.hpp>

#include <array>
#include <cmath>
#include <numbers>
#include <print>
#include <string_view>

using namespace boost::ut;

namespace {
constexpr uint32_t ModeCount{18};
constexpr double Omega{2 * std::numbers::pi * 20};
constexpr double Shift{Omega * Omega};

struct Case {
    std::string_view Geometry;
    uint32_t Resolution;
    double PoissonRatio, FictitiousScale;
    dvec3 GridOffset;
};

suite RobustnessTests = [] {
    "wide tapered-key eigenpairs certify with the assembled fallback"_test = [] {
        constexpr uint32_t count{256};
        const auto geometry = finite_cell_benchmark::MakeGeometry("tapered-key");
        const auto domain = modal::MakeTriangleSurfaceDomain(geometry.Boundary.Points, geometry.Boundary.Triangles);
        const auto operation = modal::BuildFiniteCellOperator(
            domain, {.Density = 2700, .YoungModulus = 7.2e10, .PoissonRatio = 0.19},
            {
                .Cells = finite_cell_benchmark::GridResolution(geometry, 8),
                .CutDepth = 2,
                .FictitiousScale = 1e-8,
                .PaddingCells = 0.25,
            }
        );
        const auto p1 = operation.AssembleP1Lower();
        Eigen::MatrixXd rhs(p1.Mass.rows(), 8);
        for (Eigen::Index column = 0; column < rhs.cols(); ++column)
            for (Eigen::Index row = 0; row < rhs.rows(); ++row)
                rhs(row, column) = std::sin(double(1 + row + 17 * column));
        Eigen::MatrixXd first_solve(rhs.rows(), rhs.cols()), repeated_solve(rhs.rows(), rhs.cols());
        double first_factor_seconds{}, first_solve_seconds{}, repeated_factor_seconds{}, repeated_solve_seconds{};
        modal::finite_cell::AccelerateShiftInvert first_inverse{p1.Stiffness, p1.Mass, first_factor_seconds, first_solve_seconds};
        first_inverse.set_shift(-Shift);
        first_inverse.solve_panel(rhs.data(), first_solve.data(), int(rhs.cols()));
        modal::finite_cell::AccelerateShiftInvert repeated_inverse{p1.Stiffness, p1.Mass, repeated_factor_seconds, repeated_solve_seconds};
        repeated_inverse.set_shift(-Shift);
        repeated_inverse.solve_panel(rhs.data(), repeated_solve.data(), int(rhs.cols()));
        expect(bool((first_solve.array() == repeated_solve.array()).all()));
        const auto result = modal::SolveFiniteCellEigenpairs(operation, count, Shift, 1e-8, 100);
        const auto repeated = modal::SolveFiniteCellEigenpairs(operation, count, Shift, 1e-8, 100);
        expect(result.Eigenvalues.size() == Eigen::Index(count));
        expect(repeated.Eigenvalues.size() == Eigen::Index(count));
        expect(result.Profile.FailedFactorFreeIterations > 0_u);
        expect(result.Profile.FailedFactorFreeIterations < 100_u);
        expect(result.Profile.FailedFactorFreeStagnated);
        expect(repeated.Profile.FailedFactorFreeIterations > 0_u);
        expect(repeated.Profile.FailedFactorFreeIterations < 100_u);
        expect(repeated.Profile.FailedFactorFreeStagnated);
        expect(result.Profile.FailedFactorFreeIterations == repeated.Profile.FailedFactorFreeIterations);
        if (result.Eigenvalues.size() != count || repeated.Eigenvalues.size() != count) return;
        const auto result_certification = modal::finite_cell::CertifyEigenpairs(operation, result.Eigenvalues, result.Eigenvectors);
        const auto repeated_certification = modal::finite_cell::CertifyEigenpairs(operation, repeated.Eigenvalues, repeated.Eigenvectors);
        const double residual = result_certification.RelativeResiduals.tail(count - 6).maxCoeff();
        const double repeated_residual = repeated_certification.RelativeResiduals.tail(count - 6).maxCoeff();
        const double spectrum = (repeated.Eigenvalues.tail(count - 6) - result.Eigenvalues.tail(count - 6)).norm() /
            result.Eigenvalues.tail(count - 6).norm();
        const auto shapes = finite_cell_benchmark::CompareSameDiscretizationModeShapes(
            operation, result.Eigenvalues, result.Eigenvectors, repeated.Eigenvectors
        );
        std::println(
            "wide tapered-key dofs={} modes={} fallback_iterations={}/{} spectrum={:.3e} residual={:.3e}/{:.3e} orthogonality={:.3e}/{:.3e} cluster_mac={:.6f}",
            operation.Dofs(), count, result.Profile.FailedFactorFreeIterations,
            repeated.Profile.FailedFactorFreeIterations, spectrum, residual, repeated_residual,
            result_certification.MassOrthogonalityError, repeated_certification.MassOrthogonalityError,
            shapes.ClusterMacMinimum
        );
        expect(residual < 1e-8);
        expect(repeated_residual < 1e-8);
        expect(result_certification.MassOrthogonalityError < 1e-9);
        expect(repeated_certification.MassOrthogonalityError < 1e-9);
        expect(spectrum == 0.0);
        expect((result.Eigenvectors - repeated.Eigenvectors).squaredNorm() == 0.0);
        expect((result_certification.RelativeResiduals - repeated_certification.RelativeResiduals).squaredNorm() == 0.0);
        expect(result_certification.MassOrthogonalityError == repeated_certification.MassOrthogonalityError);
        expect(shapes.ClusterMacMinimum > 0.99999);
    };

    "factor-free torus solve stays within the iteration cap"_test = [] {
        constexpr uint32_t count{128};
        const auto geometry = finite_cell_benchmark::MakeGeometry("torus");
        const auto domain = modal::MakeTriangleSurfaceDomain(geometry.Boundary.Points, geometry.Boundary.Triangles);
        const auto operation = modal::BuildFiniteCellOperator(
            domain, {.Density = 2700, .YoungModulus = 7.2e10, .PoissonRatio = 0.19},
            {
                .Cells = finite_cell_benchmark::GridResolution(geometry, 6),
                .CutDepth = 2,
                .FictitiousScale = 1e-8,
                .PaddingCells = 0.25,
            }
        );
        const auto result = modal::SolveFiniteCellEigenpairs(operation, count, Shift, 1e-8, 256);
        const auto repeated = modal::SolveFiniteCellEigenpairs(operation, count, Shift, 1e-8, 256);
        expect(result.Eigenvalues.size() == Eigen::Index(count));
        expect(repeated.Eigenvalues.size() == Eigen::Index(count));
        expect(result.Profile.FailedFactorFreeIterations == 0_u || result.Profile.FailedFactorFreeIterations < 100_u);
        if (result.Profile.FailedFactorFreeIterations) expect(result.Profile.FailedFactorFreeStagnated);
        expect(repeated.Profile.FailedFactorFreeIterations == 0_u || repeated.Profile.FailedFactorFreeIterations < 100_u);
        if (repeated.Profile.FailedFactorFreeIterations) expect(repeated.Profile.FailedFactorFreeStagnated);
        expect(result.Profile.FailedFactorFreeIterations == repeated.Profile.FailedFactorFreeIterations);
        if (result.Eigenvalues.size() != count || repeated.Eigenvalues.size() != count) return;
        const auto result_certification = modal::finite_cell::CertifyEigenpairs(operation, result.Eigenvalues, result.Eigenvectors);
        const auto repeated_certification = modal::finite_cell::CertifyEigenpairs(operation, repeated.Eigenvalues, repeated.Eigenvectors);
        const double spectrum = (repeated.Eigenvalues.tail(count - 6) - result.Eigenvalues.tail(count - 6)).norm() /
            result.Eigenvalues.tail(count - 6).norm();
        const auto shapes = finite_cell_benchmark::CompareSameDiscretizationModeShapes(
            operation, result.Eigenvalues, result.Eigenvectors, repeated.Eigenvectors
        );
        std::println(
            "wide torus dofs={} modes={} fallback_iterations={}/{} spectrum={:.3e} cluster_mac={:.6f}",
            operation.Dofs(), count, result.Profile.FailedFactorFreeIterations,
            repeated.Profile.FailedFactorFreeIterations, spectrum, shapes.ClusterMacMinimum
        );
        expect(result_certification.RelativeResiduals.tail(count - 6).maxCoeff() < 1e-8);
        expect(repeated_certification.RelativeResiduals.tail(count - 6).maxCoeff() < 1e-8);
        expect(result_certification.MassOrthogonalityError < 1e-9);
        expect(repeated_certification.MassOrthogonalityError < 1e-9);
        expect(spectrum == 0.0);
        expect((result.Eigenvectors - repeated.Eigenvectors).squaredNorm() == 0.0);
        expect((result_certification.RelativeResiduals - repeated_certification.RelativeResiduals).squaredNorm() == 0.0);
        expect(result_certification.MassOrthogonalityError == repeated_certification.MassOrthogonalityError);
        expect(shapes.ClusterMacMinimum > 0.99999);
    };

    "exact fallback refines marginal physical modes"_test = [] {
        const auto geometry = finite_cell_benchmark::MakeGeometry("bar");
        const auto domain = modal::MakeTriangleSurfaceDomain(geometry.Boundary.Points, geometry.Boundary.Triangles);
        const auto operation = modal::BuildFiniteCellOperator(
            domain, {.Density = 2700, .YoungModulus = 7.2e10, .PoissonRatio = 0.19},
            {
                .Cells = finite_cell_benchmark::GridResolution(geometry, 48),
                .CutDepth = 2,
                .FictitiousScale = 1e-8,
                .PaddingCells = 0.25,
            }
        );
        const auto result = modal::finite_cell::SolveAssembledEigenpairs(operation, ModeCount, Shift, 1e-8, 100);
        expect(result.Eigenvalues.size() == Eigen::Index(ModeCount));
        if (result.Eigenvalues.size() != ModeCount) return;
        const auto result_certification = modal::finite_cell::CertifyEigenpairs(operation, result.Eigenvalues, result.Eigenvectors);
        const double residual = result_certification.RelativeResiduals.tail(ModeCount - 6).maxCoeff();
        std::println(
            "exact fallback refinement dofs={} residual={:.3e} orthogonality={:.3e}",
            operation.Dofs(), residual, result_certification.MassOrthogonalityError
        );
        expect(residual < 1e-9);
        expect(result_certification.MassOrthogonalityError < 1e-9);
    };

    "finite-cell production solve meets conditioning and registration gates"_test = [] {
        constexpr std::array cases{
            Case{"thin-plate", 9, 0.19, 1e-10, {0.20, -0.15, 0.10}},
            Case{"l-bracket", 9, 0.33, 1e-6, {-0.20, 0.15, -0.10}},
            Case{"cup", 8, 0.45, 1e-8, {0.15, 0.20, -0.15}},
            Case{"narrow-waist", 8, 0.33, 1e-10, {-0.15, -0.20, 0.15}},
            Case{"ellipsoid", 10, 0.45, 1e-6, {0.20, -0.10, -0.20}},
        };
        for (const auto &entry : cases) {
            const auto geometry = finite_cell_benchmark::MakeGeometry(entry.Geometry);
            const auto domain = modal::MakeTriangleSurfaceDomain(geometry.Boundary.Points, geometry.Boundary.Triangles);
            const AcousticMaterialProperties material{.Density = 2700, .YoungModulus = 7.2e10, .PoissonRatio = entry.PoissonRatio};
            const auto operation = modal::BuildFiniteCellOperator(
                domain, material,
                {
                    .Cells = finite_cell_benchmark::GridResolution(geometry, entry.Resolution),
                    .CutDepth = 2,
                    .FictitiousScale = entry.FictitiousScale,
                    .PaddingCells = 0.25,
                    .GridOffsetCells = entry.GridOffset,
                }
            );
            const auto assembled = modal::finite_cell::SolveAssembledEigenpairs(operation, ModeCount, Shift, 1e-9, 1000);
            const auto production = modal::SolveFiniteCellEigenpairs(operation, ModeCount, Shift, 1e-8, 300);
            expect(assembled.Eigenvalues.size() == Eigen::Index(ModeCount)) << entry.Geometry;
            expect(production.Eigenvalues.size() == Eigen::Index(ModeCount)) << entry.Geometry;
            expect(production.Profile.FailedFactorFreeIterations == 0_u) << entry.Geometry;
            if (assembled.Eigenvalues.size() != ModeCount || production.Eigenvalues.size() != ModeCount) continue;
            const auto assembled_certification = modal::finite_cell::CertifyEigenpairs(operation, assembled.Eigenvalues, assembled.Eigenvectors);
            const auto production_certification = modal::finite_cell::CertifyEigenpairs(operation, production.Eigenvalues, production.Eigenvectors);
            const double assembled_residual = assembled_certification.RelativeResiduals.tail(ModeCount - 6).maxCoeff();
            expect(assembled_residual < 1e-8) << entry.Geometry << assembled_residual;
            expect(assembled_certification.MassOrthogonalityError < 1e-9) << entry.Geometry << assembled_certification.MassOrthogonalityError;
            const double spectrum = (production.Eigenvalues.tail(ModeCount - 6) - assembled.Eigenvalues.tail(ModeCount - 6)).norm() /
                assembled.Eigenvalues.tail(ModeCount - 6).norm();
            const double residual = production_certification.RelativeResiduals.tail(ModeCount - 6).maxCoeff();
            const auto shapes = finite_cell_benchmark::CompareSameDiscretizationModeShapes(
                operation, assembled.Eigenvalues, assembled.Eigenvectors, production.Eigenvectors
            );
            std::println(
                "robustness geometry={} dofs={} cut={} nu={:.2f} fictitious={:.0e} offset={:.2f}/{:.2f}/{:.2f} iterations={} spectrum={:.3e} residual={:.3e} orthogonality={:.3e} cluster_mac={:.6f}",
                entry.Geometry, operation.Dofs(), operation.Profile.CutCells, entry.PoissonRatio, entry.FictitiousScale,
                entry.GridOffset.x, entry.GridOffset.y, entry.GridOffset.z, production.Iterations, spectrum, residual,
                production_certification.MassOrthogonalityError, shapes.ClusterMacMinimum
            );
            expect(spectrum < 1e-9) << entry.Geometry;
            expect(residual < 1e-8) << entry.Geometry;
            expect(production_certification.MassOrthogonalityError < 1e-9) << entry.Geometry;
            expect(shapes.ClusterMacMinimum > 0.99999) << entry.Geometry;
            if (entry.Geometry == "cup") {
                const auto fallback = modal::SolveFiniteCellEigenpairs(operation, ModeCount, Shift, 1e-8, 12);
                expect(fallback.Profile.FailedFactorFreeIterations > 0_u);
                const auto fallback_certification = modal::finite_cell::CertifyEigenpairs(operation, fallback.Eigenvalues, fallback.Eigenvectors);
                expect(fallback_certification.RelativeResiduals.tail(ModeCount - 6).maxCoeff() < 1e-8);
                expect(fallback_certification.MassOrthogonalityError < 1e-9);
            }
        }
    };
};
} // namespace

int main() { return RunSuites(); }
