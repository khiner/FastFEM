#pragma once

#include "audio/FiniteCell.h"
#include "audio/Tet10Assembler.h"
#include "mesh/TetMesh.h"

#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace finite_cell_benchmark {
struct ModeShapeComparison {
    uint32_t Samples{}, Clusters{}, LargestCluster{};
    double PairedMacMinimum{}, BestMacMinimum{}, ClusterMacMinimum{};
};

using InterpolationStencil = modal::FiniteCellOperator::InterpolationStencil;

inline std::optional<InterpolationStencil> FiniteCellStencil(const modal::FiniteCellOperator &operation, const dvec3 &point) {
    return operation.InterpolationAt(point);
}

inline std::optional<InterpolationStencil> Tet10Stencil(
    const TetMesh &mesh, const modal::Tet10Assembler &operation, const dvec3 &point
) {
    static constexpr uint32_t edge_corners[6][2]{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
    for (uint32_t element_index = 0; element_index < operation.Elements().size(); ++element_index) {
        const auto &element = operation.Elements()[element_index];
        const auto &tet = mesh.Tets[element_index];
        const dvec3 delta = point - mesh.Points[tet[0]];
        std::array<double, 4> barycentric{};
        bool inside = true;
        for (uint32_t corner = 0; corner < 4; ++corner) {
            barycentric[corner] = (corner == 0 ? 1.0 : 0.0) +
                element.Gradients[corner][0] * delta.x + element.Gradients[corner][1] * delta.y +
                element.Gradients[corner][2] * delta.z;
            inside &= barycentric[corner] >= -1e-10 && barycentric[corner] <= 1 + 1e-10;
        }
        if (!inside) continue;
        InterpolationStencil result{.Count = 10};
        for (uint32_t corner = 0; corner < 4; ++corner) {
            result.Nodes[corner] = element.Nodes[corner];
            result.Weights[corner] = barycentric[corner] * (2 * barycentric[corner] - 1);
        }
        for (uint32_t edge = 0; edge < 6; ++edge) {
            result.Nodes[4 + edge] = element.Nodes[4 + edge];
            result.Weights[4 + edge] = 4 * barycentric[edge_corners[edge][0]] * barycentric[edge_corners[edge][1]];
        }
        return result;
    }
    return std::nullopt;
}

inline Eigen::MatrixXd SampleModes(
    const std::vector<InterpolationStencil> &stencils, const Eigen::MatrixXd &modes, uint32_t first_mode
) {
    Eigen::MatrixXd result(3 * stencils.size(), modes.cols() - first_mode);
    result.setZero();
    for (uint32_t sample = 0; sample < stencils.size(); ++sample)
        for (uint32_t node = 0; node < stencils[sample].Count; ++node)
            for (uint32_t component = 0; component < 3; ++component)
                for (Eigen::Index mode = first_mode; mode < modes.cols(); ++mode)
                    result(3 * sample + component, mode - first_mode) +=
                        stencils[sample].Weights[node] * modes(3 * stencils[sample].Nodes[node] + component, mode);
    return result;
}

inline ModeShapeComparison CompareSameDiscretizationModeShapes(
    const modal::FiniteCellOperator &operation, const Eigen::VectorXd &reference_values,
    const Eigen::MatrixXd &reference_modes, const Eigen::MatrixXd &candidate_modes, uint32_t first_mode = 6,
    uint32_t accepted_modes = 0
) {
    ModeShapeComparison result{.Samples = uint32_t(operation.Nodes.size())};
    if (reference_modes.rows() != candidate_modes.rows() || reference_modes.cols() != candidate_modes.cols() ||
        reference_modes.cols() <= first_mode || reference_values.size() != reference_modes.cols())
        return result;
    Eigen::MatrixXd reference_mass(reference_modes.rows(), reference_modes.cols());
    Eigen::MatrixXd candidate_mass(candidate_modes.rows(), candidate_modes.cols());
    operation.ApplyMass(reference_modes.data(), reference_mass.data(), uint32_t(reference_modes.cols()));
    operation.ApplyMass(candidate_modes.data(), candidate_mass.data(), uint32_t(candidate_modes.cols()));
    Eigen::MatrixXd reference = reference_modes.rightCols(reference_modes.cols() - first_mode);
    Eigen::MatrixXd candidate = candidate_modes.rightCols(candidate_modes.cols() - first_mode);
    Eigen::MatrixXd reference_action = reference_mass.rightCols(reference_mass.cols() - first_mode);
    Eigen::MatrixXd candidate_action = candidate_mass.rightCols(candidate_mass.cols() - first_mode);
    for (Eigen::Index mode = 0; mode < reference.cols(); ++mode) {
        const double reference_norm = std::sqrt(std::abs(reference.col(mode).dot(reference_action.col(mode))));
        const double candidate_norm = std::sqrt(std::abs(candidate.col(mode).dot(candidate_action.col(mode))));
        reference.col(mode) /= reference_norm;
        candidate.col(mode) /= candidate_norm;
        reference_action.col(mode) /= reference_norm;
        candidate_action.col(mode) /= candidate_norm;
    }
    const Eigen::MatrixXd overlap = reference.transpose() * candidate_action;
    const Eigen::MatrixXd mac = overlap.array().square();
    const Eigen::Index accepted_end = accepted_modes ? std::min<Eigen::Index>(accepted_modes, reference_values.size()) : reference_values.size();
    const Eigen::Index accepted_count = accepted_end - first_mode;
    if (accepted_count <= 0) return result;
    result.PairedMacMinimum = mac.diagonal().head(accepted_count).minCoeff();
    result.BestMacMinimum = std::numeric_limits<double>::infinity();
    for (Eigen::Index mode = 0; mode < accepted_count; ++mode)
        result.BestMacMinimum = std::min(result.BestMacMinimum, mac.row(mode).maxCoeff());
    result.ClusterMacMinimum = 1;
    for (Eigen::Index first = first_mode; first < accepted_end;) {
        Eigen::Index end = first + 1;
        while (end < reference_values.size()) {
            const double scale = std::max({std::abs(reference_values[end - 1]), std::abs(reference_values[end]), 1.0});
            if ((reference_values[end] - reference_values[end - 1]) / scale >= 0.02) break;
            ++end;
        }
        const Eigen::Index width = end - first, offset = first - first_mode;
        const Eigen::JacobiSVD<Eigen::MatrixXd> svd(overlap.block(offset, offset, width, width));
        result.ClusterMacMinimum = std::min(result.ClusterMacMinimum, std::pow(svd.singularValues().minCoeff(), 2));
        ++result.Clusters;
        result.LargestCluster = std::max(result.LargestCluster, uint32_t(width));
        first = end;
    }
    return result;
}
} // namespace finite_cell_benchmark
