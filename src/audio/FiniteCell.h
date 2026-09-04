#pragma once

#include "AcousticMaterialProperties.h"
#include "numeric/vec3.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace modal {
enum struct DomainRegion { Inside,
                           Outside,
                           Cut };

// Signed distances are negative inside and positive outside.
// Min and Max tightly bound the physical domain before PaddingCells expands the background grid.
// Triangle surfaces must be watertight and non-self-intersecting.
struct ImplicitDomain {
    dvec3 Min{}, Max{};
    std::function<double(const dvec3 &)> SignedDistance;
    std::function<DomainRegion(const dvec3 &center, const dvec3 &half)> ClassifyBox;
};

ImplicitDomain MakeBoxDomain(dvec3 min, dvec3 max);
ImplicitDomain MakeSphereDomain(dvec3 center, double radius);
ImplicitDomain MakeSphericalShellDomain(dvec3 center, double inner_radius, double outer_radius);
ImplicitDomain MakeCylinderDomain(dvec3 center, double radius, double length);
ImplicitDomain MakeTriangleSurfaceDomain(std::span<const dvec3> points, std::span<const uint32_t> triangle_indices);

struct FiniteCellConfig {
    uvec3 Cells{12, 12, 12};
    uint32_t CutDepth{3};
    double FictitiousScale{1e-8};
    double PaddingCells{0.25};
    dvec3 GridOffsetCells{};
};

struct FiniteCellProfile {
    double Assemble{}, PhysicalVolume{}, MomentFitMaximumResidual{};
    uint32_t BackgroundCells{}, ActiveCells{}, CutCells{}, MomentFittedCells{}, MomentFitFallbackCells{}, Dofs{};
    uint64_t QuadraturePoints{}, UncompressedQuadraturePoints{};
};

struct FiniteCellOperator {
    static constexpr uint32_t NodesPerCell{27};

    struct P1Stencil {
        std::array<uint32_t, 8> Nodes{};
        std::array<double, 8> Weights{};
        uint8_t Count{};
    };

    struct QuadraturePoint {
        dvec3 Reference{};
        double Weight{};
        uint8_t Fictitious{};
    };

    struct InterpolationStencil {
        std::array<uint32_t, NodesPerCell> Nodes{};
        std::array<double, NodesPerCell> Weights{};
        uint8_t Count{NodesPerCell};
    };

    struct Cell {
        std::array<uint32_t, NodesPerCell> Nodes{};
        dvec3 InverseHalf{};
        uint32_t QuadratureOffset{}, QuadratureCount{}, UncompressedQuadratureCount{};
        uint8_t Color{}, Cut{}, MomentFitted{}, MomentFitFallback{};
    };

    struct AssembledLower {
        Eigen::SparseMatrix<double> Mass;
        Eigen::SparseMatrix<double> Stiffness;
    };

    struct PackedCutOperators {
        std::vector<uint32_t> InteriorCells, CutCells;
        std::vector<double> Mass, Shifted;
        std::vector<double> MassCells, ShiftedCells;
        double Alpha{}, BuildSeconds{};
    };

    std::vector<Cell> Cells;
    std::vector<QuadraturePoint> Quadrature;
    std::vector<dvec3> Nodes;
    std::vector<P1Stencil> P1Stencils;
    std::vector<uint32_t> NodeOccurrenceOffsets;
    std::vector<uint32_t> NodeOccurrences;
    std::vector<int32_t> CellAtBackgroundIndex;
    FiniteCellProfile Profile;
    uvec3 GridCells{};
    dvec3 GridMin{}, CellStep{};
    uint32_t NumP1Nodes{};
    double Density{}, Lambda{}, Mu{}, FictitiousScale{};

    uint32_t Dofs() const { return 3 * uint32_t(Nodes.size()); }
    std::optional<InterpolationStencil> InterpolationAt(dvec3 point) const;
    void ApplyMass(const double *input, double *output, uint32_t width) const;
    void ApplyMassShifted(
        const double *input, double *mass_output, double *shifted_output, uint32_t width, double alpha
    ) const;
    void ApplyMassShiftedExpandedPackedCut(
        PackedCutOperators &, const double *input, double *mass_output, double *shifted_output, uint32_t width
    ) const;
    void ApplyStiffness(const double *input, double *output, uint32_t width) const;
    void ApplyShifted(const double *input, double *output, uint32_t width, double alpha) const;
    void RestrictP1(const double *input, double *output, uint32_t width) const;
    void ProlongP1(const double *input, double *output, uint32_t width) const;
    Eigen::VectorXd ShiftedDiagonal(double alpha) const;
    // Writes the lower triangle of cell `K + alpha*M` in row-packed order over `3 * NodesPerCell` rows.
    void PackCellShiftedLower(uint32_t cell, double alpha, std::span<double> packed) const;
    PackedCutOperators BuildPackedCutOperators(double alpha) const;
    PackedCutOperators BuildPackedCutOperators(double alpha, std::span<const double> packed_shifted_elements) const;
    FiniteCellOperator WithFictitiousScale(double scale) const;
    AssembledLower AssembleLower() const;
    AssembledLower AssembleP1Lower() const;
    Eigen::SparseMatrix<double> AssembleP1ShiftedLower(double alpha) const;
};

struct FiniteCellCertification {
    Eigen::VectorXd RelativeResiduals;
    double MassOrthogonalityError{};
};

FiniteCellOperator BuildFiniteCellOperator(const ImplicitDomain &, const AcousticMaterialProperties &, FiniteCellConfig = {});
FiniteCellCertification CertifyFiniteCellEigenpairs(
    const FiniteCellOperator &, const Eigen::VectorXd &eigenvalues, const Eigen::MatrixXd &eigenvectors
);
} // namespace modal
