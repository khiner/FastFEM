#pragma once

#include "numeric/dvec3.h"

#include "numeric/quat.h"
#include "numeric/vec3.h"
#include <array>
#include <compare>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace fastfem {
using numeric::dvec3, numeric::quat, numeric::vec3;

struct SolveMonitor;

struct AcousticMaterialProperties {
    double Density{}, YoungModulus{}, PoissonRatio{};
    double Alpha{}, Beta{};
    double Lambda() const { return (PoissonRatio * YoungModulus) / ((1 + PoissonRatio) * (1 - 2 * PoissonRatio)); }
    double Mu() const { return YoungModulus / (2 * (1 + PoissonRatio)); }
    auto operator<=>(const AcousticMaterialProperties &) const = default;
};

enum struct Discretization { Tet10,
                             FiniteCell };

struct SolverConfig {
    float MinModeFreq{20};
    float MaxModeFreq{16'000};
    uint32_t NumModes{30};
    uint32_t NumFemModes{45};
    double Tolerance{1e-8};
    uint32_t MaxRestarts{100};
    std::optional<float> FundamentalFreq{};
};

enum class TetRefinement { None,
                           Quality,
                           QualityAndResolution };

struct TetrahedralizationConfig {
    TetRefinement Refinement{TetRefinement::None};
    std::vector<dvec3> Holes;
};

struct FiniteCellConfig {
    uint32_t CutDepth{3};
    double FictitiousScale{1e-8};
    double PaddingCells{0.25};
    dvec3 GridOffsetCells{};
};

struct SurfaceSolveConfig {
    SolverConfig Modal{};
    TetrahedralizationConfig Tetrahedralization{};
    FiniteCellConfig FiniteCell{};
    // Divisions along the longest input axis. Sets finite-cell spacing and, for
    // QualityAndResolution, Tet10 surface edge lengths and tetrahedron volume targets.
    uint32_t Resolution{12};
    float SurfaceSimplificationRatio{1};
};

struct ModalModes {
    std::vector<float> Freqs;
    std::vector<float> T60s;
    std::vector<std::vector<vec3>> Shapes;
    std::vector<vec3> Positions;
    float OriginalFundamentalFreq{Freqs.empty() ? 0 : Freqs.front()};
    vec3 BakedScale{1, 1, 1};
    bool operator==(const ModalModes &) const = default;
};

struct MassProperties {
    double Mass{};
    vec3 CenterOfMass{};
    vec3 InertiaDiagonal{};
    quat InertiaOrientation{};
    bool operator==(const MassProperties &) const = default;
};

struct ModalEigenSummary {
    std::vector<double> Eigenvalues;
    std::vector<std::vector<vec3>> Shapes;
    AcousticMaterialProperties SolvedMaterial{};
    bool operator==(const ModalEigenSummary &) const = default;
};

struct TetMesh {
    std::vector<dvec3> Points;
    std::vector<std::array<uint32_t, 4>> Tets;
    bool operator==(const TetMesh &) const = default;
};

struct ModeBasis {
    struct Storage;
    std::shared_ptr<const Storage> Data;

    explicit operator bool() const { return bool(Data); }
};

struct SolveReuse {
    const ModeBasis *SeedBasis{};
    bool KeepBasis{};
};

struct ModalResult {
    ModalModes Modes;
    MassProperties Mass;
    ModalEigenSummary Summary;
    ModeBasis Basis;
    std::vector<uint32_t> SamplePointOfExcitation;
    TetMesh Tetrahedra;
};

std::expected<ModalResult, std::string> Surface2Modes(
    std::span<const vec3> positions, std::span<const uint32_t> triangle_indices,
    const AcousticMaterialProperties &, std::span<const vec3> excitation_positions,
    vec3 baked_scale, Discretization, SurfaceSolveConfig = {}, SolveReuse = {}, SolveMonitor * = nullptr
);

std::optional<ModalModes> RescaleModes(
    const ModalEigenSummary &, const ModalModes &current,
    const AcousticMaterialProperties &, SolverConfig = {}
);
} // namespace fastfem
