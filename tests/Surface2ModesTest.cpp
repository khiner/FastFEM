#include "FastFEM/Surface2Modes.h"
#include "FastFEM/SolveMonitor.h"
#include "ModalTestGeometry.h"

#include <boost/ut.hpp>

#include <cmath>
#include <vector>

using namespace boost::ut;

int main() {
    "surface discretizations produce modal models"_test = [] {
        constexpr fastfem::AcousticMaterialProperties material{
            .Density = 1000,
            .YoungModulus = 1e7,
            .PoissonRatio = 0.2,
            .Alpha = 1,
            .Beta = 1e-7,
        };
        const auto box = modal_test::AxisBarSurface();
        const std::vector<vec3> positions(box.Points.begin(), box.Points.end());
        const fastfem::SurfaceSolveConfig config{
            .Modal = {
                .MinModeFreq = 1,
                .MaxModeFreq = 100'000,
                .NumModes = 4,
                .NumFemModes = 12,
                .Tolerance = 1e-8,
                .MaxRestarts = 150,
            },
            .FiniteCell = {
                .CutDepth = 3,
                .PaddingCells = 0,
            },
        };
        for (const auto discretization : {fastfem::Discretization::Tet10, fastfem::Discretization::FiniteCell}) {
            fastfem::SolveMonitor monitor;
            const auto result = fastfem::Surface2Modes(positions, box.Triangles, material, positions, vec3{1}, discretization, config, {.KeepBasis = true}, &monitor);
            expect(bool(result)) << (result ? "" : result.error());
            if (!result) continue;
            expect(!result->Modes.Freqs.empty());
            expect(result->Modes.Freqs.size() == result->Modes.T60s.size());
            expect(result->Modes.Shapes.size() == positions.size());
            expect(result->Modes.Positions.size() == positions.size());
            expect(result->Summary.Eigenvalues.size() == config.Modal.NumFemModes);
            expect(result->Summary.Shapes.size() == positions.size());
            expect(result->SamplePointOfExcitation.size() == positions.size());
            expect(bool(result->Basis) == (discretization == fastfem::Discretization::Tet10));
            expect(result->Modes.BakedScale == vec3{1});
            expect((discretization == fastfem::Discretization::Tet10) == !result->Tetrahedra.Tets.empty());
            expect(monitor.Progress.load(std::memory_order_relaxed) == 1.f);
            expect(monitor.Stage.load(std::memory_order_relaxed) == fastfem::SolveStage::Complete);
            const double expected_mass = material.Density * modal_test::BarExtent.x * modal_test::BarExtent.y * modal_test::BarExtent.z;
            expect(std::abs(result->Mass.Mass / expected_mass - 1) < 0.01) << result->Mass.Mass;
        }
    };

    "tetrahedral refinement modes control size explicitly"_test = [] {
        const auto box = modal_test::AxisBarSurface();
        std::vector<fastfem::Vec3> positions(box.Points.begin(), box.Points.end());
        fastfem::SurfaceSolveConfig config{
            .Modal = {.MinModeFreq = 1, .MaxModeFreq = 100'000, .NumModes = 4, .NumFemModes = 12, .MaxRestarts = 150},
            .Tetrahedralization = {.Refinement = fastfem::TetRefinement::QualityAndResolution},
            .Resolution = 3,
        };
        const auto solve = [&] {
            return fastfem::Surface2Modes(
                positions, box.Triangles,
                {.Density = 1000, .YoungModulus = 1e7, .PoissonRatio = 0.2},
                positions, {1, 1, 1}, fastfem::Discretization::Tet10, config
            );
        };
        const auto refined = solve();
        expect(bool(refined)) << (refined ? "" : refined.error());
        if (refined) {
            expect(refined->Tetrahedra.Tets.size() > 6u);
            double volume{};
            for (const auto &tet : refined->Tetrahedra.Tets) {
                const auto &points = refined->Tetrahedra.Points;
                const double tet_volume = std::abs(numeric::Dot(points[tet[1]] - points[tet[0]], numeric::Cross(points[tet[2]] - points[tet[0]], points[tet[3]] - points[tet[0]]))) / 6;
                volume += tet_volume;
            }
            expect(std::abs(volume / modal_test::Volume(box) - 1) < 1e-6);
            for (auto &point : positions) point *= 2;
            const auto scaled = solve();
            expect(bool(scaled));
            if (scaled) {
                expect(scaled->Tetrahedra.Tets.size() == refined->Tetrahedra.Tets.size());
                expect(std::abs(scaled->Mass.Mass / refined->Mass.Mass - 8) < 1e-5);
                expect(std::abs(scaled->Modes.Freqs.front() / refined->Modes.Freqs.front() - 0.5) < 1e-5);
            }
        }
        config.Resolution = 0;
        const auto invalid = solve();
        expect(!invalid);
        if (!invalid) expect(invalid.error().find("resolution") != std::string::npos);
    };
}
