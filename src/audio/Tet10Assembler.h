#pragma once

#include "AcousticMaterialProperties.h"

#include <Eigen/SparseCore>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

struct TetMesh;

namespace modal {
struct Tet10Assembler {
    static constexpr uint32_t NodesPerElement{10};

    struct AssembledLower {
        Eigen::SparseMatrix<double> Mass;
        Eigen::SparseMatrix<double> Stiffness;
    };

    struct Element {
        std::array<uint32_t, NodesPerElement> Nodes;
        double Volume;
        std::array<std::array<double, 3>, 4> Gradients;
    };

    struct Topology {
        std::vector<Element> Elements;
        uint32_t NumNodes{};
    };

    std::shared_ptr<const Topology> State;
    uint32_t NumNodes{};
    double Density{}, Lambda{}, Mu{};

    Tet10Assembler(const TetMesh &, const AcousticMaterialProperties &);
    Tet10Assembler(std::shared_ptr<const Topology>, const AcousticMaterialProperties &);

    static std::shared_ptr<const Topology> BuildTopology(const TetMesh &);
    const std::vector<Element> &Elements() const { return State->Elements; }

    uint32_t Dofs() const { return 3 * NumNodes; }
    AssembledLower AssembleLower() const;
};
} // namespace modal
