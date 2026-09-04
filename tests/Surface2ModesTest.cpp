#include "audio/Surface2Modes.h"
#include "FastFEM/SolveMonitor.h"
#include "FastFEM/Surface2Modes.h"
#include "FiniteCellBenchmarkGeometry.h"

#include <boost/ut.hpp>

#include <cmath>
#include <vector>

using namespace boost::ut;

int main() {
    "surface discretizations produce modal models"_test = [] {
        constexpr AcousticMaterialProperties material{
            .Density = 1000,
            .YoungModulus = 1e7,
            .PoissonRatio = 0.2,
            .Alpha = 1,
            .Beta = 1e-7,
        };
        const auto box = finite_cell_benchmark::AxisBarSurface();
        std::vector<vec3> positions;
        positions.reserve(box.Points.size());
        for (const auto point : box.Points) positions.emplace_back(point);
        const modal::SurfaceSolveConfig config{
            .Modal = {
                .MinModeFreq = 1,
                .MaxModeFreq = 100'000,
                .NumModes = 4,
                .NumFemModes = 12,
                .Tolerance = 1e-8,
                .MaxRestarts = 150,
            },
            .FiniteCell = {
                .Cells = {8, 4, 3},
                .CutDepth = 3,
            },
        };
        for (const auto discretization : {modal::Discretization::Tet10, modal::Discretization::FiniteCell}) {
            const auto result = modal::Surface2Modes(positions, box.Triangles, material, positions, vec3{1}, discretization, config, {.KeepBasis = true});
            expect(bool(result)) << (result ? "" : result.error());
            if (!result) continue;
            expect(!result->Modes.Freqs.empty());
            expect(result->Modes.Freqs.size() == result->Modes.T60s.size());
            expect(result->Modes.Shapes.size() == positions.size());
            expect(result->Modes.Positions.size() == positions.size());
            expect(result->Summary.Eigenvalues.size() == config.Modal.NumFemModes);
            expect(result->Summary.Shapes.size() == positions.size());
            expect(result->SamplePointOfExcitation.size() == positions.size());
            if (discretization == modal::Discretization::Tet10) {
                expect(result->Basis.rows() == result->Profile.Dofs);
                expect(result->Basis.cols() == config.Modal.NumFemModes);
            } else expect(result->Basis.empty());
            expect(result->Modes.BakedScale == vec3{1});
            expect((discretization == modal::Discretization::Tet10) == !result->Tetrahedra.Tets.empty());
            const double expected_mass = material.Density * finite_cell_benchmark::BarExtent.x * finite_cell_benchmark::BarExtent.y * finite_cell_benchmark::BarExtent.z;
            expect(std::abs(result->MassProps.Mass / expected_mass - 1) < 0.01) << result->MassProps.Mass;
        }
    };

    "public surface API produces a modal result"_test = [] {
        const auto box = finite_cell_benchmark::AxisBarSurface();
        std::vector<fastfem::Vec3> positions;
        positions.reserve(box.Points.size());
        for (const auto point : box.Points) positions.push_back({float(point.x), float(point.y), float(point.z)});
        fastfem::SolveMonitor monitor;
        const auto result = fastfem::Surface2Modes(
            positions, box.Triangles,
            {.Density = 1000, .YoungModulus = 1e7, .PoissonRatio = 0.2, .Alpha = 1, .Beta = 1e-7},
            positions, {1, 1, 1}, fastfem::Discretization::Tet10,
            {.Modal = {.MinModeFreq = 1, .MaxModeFreq = 100'000, .NumModes = 4, .NumFemModes = 12, .MaxRestarts = 150}},
            {.KeepBasis = true}, &monitor
        );
        expect(bool(result)) << (result ? "" : result.error());
        if (!result) return;
        expect(!result->Modes.Freqs.empty());
        expect(bool(result->Basis));
        expect(!result->Tetrahedra.Tets.empty());
        expect(result->SamplePointOfExcitation.size() == positions.size());
        expect(monitor.Progress.load(std::memory_order_relaxed) == 1.f);
    };
}
