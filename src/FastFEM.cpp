#include "FastFEM/Surface2Modes.h"

#include "audio/Surface2Modes.h"

#include <utility>

struct fastfem::ModeBasis::Storage {
    numeric::Matrix<float> Matrix;
};

namespace {
modal::SolverConfig Import(const fastfem::SolverConfig &config) {
    return {config.MinModeFreq, config.MaxModeFreq, config.NumModes, config.NumFemModes, config.Tolerance, config.MaxRestarts, config.FundamentalFreq};
}

modal::SurfaceSolveConfig Import(const fastfem::SurfaceSolveConfig &config) {
    return {
        .Modal = Import(config.Modal),
        .Tetrahedralization = {
            .Quality = config.Tetrahedralization.Quality,
            .MaxVolume = config.Tetrahedralization.MaxVolume,
            .Holes = config.Tetrahedralization.Holes,
        },
        .FiniteCell = {
            .Cells = {config.FiniteCell.Cells[0], config.FiniteCell.Cells[1], config.FiniteCell.Cells[2]},
            .CutDepth = config.FiniteCell.CutDepth,
            .FictitiousScale = config.FiniteCell.FictitiousScale,
            .PaddingCells = config.FiniteCell.PaddingCells,
            .GridOffsetCells = config.FiniteCell.GridOffsetCells,
        },
        .SurfaceSimplificationRatio = config.SurfaceSimplificationRatio,
    };
}
} // namespace

std::expected<fastfem::ModalResult, std::string> fastfem::Surface2Modes(
    std::span<const Vec3> positions, std::span<const uint32_t> triangle_indices,
    const AcousticMaterialProperties &material, std::span<const Vec3> excitation_positions,
    Vec3 baked_scale, Discretization discretization, SurfaceSolveConfig config,
    SolveReuse reuse, SolveMonitor *monitor
) {
    const modal::SolveReuse imported_reuse{
        .SeedBasis = reuse.SeedBasis && reuse.SeedBasis->Data ? &reuse.SeedBasis->Data->Matrix : nullptr,
        .KeepBasis = reuse.KeepBasis,
    };
    auto result = modal::Surface2Modes(
        positions, triangle_indices, material, excitation_positions, baked_scale,
        discretization == Discretization::Tet10 ? modal::Discretization::Tet10 : modal::Discretization::FiniteCell,
        Import(config), imported_reuse, monitor
    );
    if (!result) return std::unexpected(std::move(result.error()));

    ModeBasis basis;
    if (!result->Basis.empty()) basis.Data = std::make_shared<ModeBasis::Storage>(std::move(result->Basis));
    return ModalResult{
        .Modes = std::move(result->Modes),
        .Mass = std::move(result->MassProps),
        .Summary = std::move(result->Summary),
        .Basis = std::move(basis),
        .SamplePointOfExcitation = std::move(result->SamplePointOfExcitation),
        .Tetrahedra = std::move(result->Tetrahedra),
    };
}

std::optional<fastfem::ModalModes> fastfem::RescaleModes(
    const ModalEigenSummary &summary, const ModalModes &current,
    const AcousticMaterialProperties &material, SolverConfig config
) {
    return modal::RescaleModes(summary, current, material, Import(config));
}
