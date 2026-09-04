#include "Tet10Assembler.h"

#include "mesh/TetMesh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_map>

namespace {
using uint = uint32_t;

constexpr uint NodesPerElement{10};
constexpr uint EdgeCorners[6][2]{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

struct BaryTerm {
    double Coeff;
    std::array<int, 4> Exp;
};
using BaryPoly = std::vector<BaryTerm>;

struct Tet10ReferenceBasis {
    double Mass[10][10];
    double Grad[10][4][10][4];
};

BaryPoly Multiply(const BaryPoly &a, const BaryPoly &b) {
    BaryPoly product;
    product.reserve(a.size() * b.size());
    for (const auto &ta : a) {
        for (const auto &tb : b) {
            product.push_back({ta.Coeff * tb.Coeff, {ta.Exp[0] + tb.Exp[0], ta.Exp[1] + tb.Exp[1], ta.Exp[2] + tb.Exp[2], ta.Exp[3] + tb.Exp[3]}});
        }
    }
    return product;
}

double UnitIntegral(const BaryPoly &p) {
    static constexpr double Factorial[]{1, 1, 2, 6, 24, 120, 720, 5040};
    double sum = 0;
    for (const auto &term : p) {
        const auto &e = term.Exp;
        sum += term.Coeff * 6 * Factorial[e[0]] * Factorial[e[1]] * Factorial[e[2]] * Factorial[e[3]] / Factorial[e[0] + e[1] + e[2] + e[3] + 3];
    }
    return sum;
}

const Tet10ReferenceBasis &ReferenceBasis() {
    static const Tet10ReferenceBasis basis = [] {
        std::array<BaryPoly, NodesPerElement> n;
        std::array<std::array<BaryPoly, 4>, NodesPerElement> dn;
        for (int i = 0; i < 4; ++i) {
            n[i] = {{2, {2 * (i == 0), 2 * (i == 1), 2 * (i == 2), 2 * (i == 3)}}, {-1, {i == 0, i == 1, i == 2, i == 3}}};
            dn[i][i] = {{4, {i == 0, i == 1, i == 2, i == 3}}, {-1, {0, 0, 0, 0}}};
        }
        for (uint edge = 0; edge < 6; ++edge) {
            const int i = EdgeCorners[edge][0], j = EdgeCorners[edge][1];
            n[4 + edge] = {{4, {i == 0 || j == 0, i == 1 || j == 1, i == 2 || j == 2, i == 3 || j == 3}}};
            dn[4 + edge][i] = {{4, {j == 0, j == 1, j == 2, j == 3}}};
            dn[4 + edge][j] = {{4, {i == 0, i == 1, i == 2, i == 3}}};
        }

        Tet10ReferenceBasis result;
        for (uint a = 0; a < NodesPerElement; ++a) {
            for (uint c = 0; c < NodesPerElement; ++c) {
                result.Mass[a][c] = UnitIntegral(Multiply(n[a], n[c]));
                for (int k = 0; k < 4; ++k) {
                    for (int l = 0; l < 4; ++l) {
                        result.Grad[a][k][c][l] = dn[a][k].empty() || dn[c][l].empty() ? 0 : UnitIntegral(Multiply(dn[a][k], dn[c][l]));
                    }
                }
            }
        }
        return result;
    }();
    return basis;
}

double Determinant(const dvec3 &a, const dvec3 &b, const dvec3 &c, const dvec3 &d) {
    return numeric::Dot(d - a, numeric::Cross(b - a, c - a));
}

std::array<std::array<double, 3>, 4> ShapeGradients(const TetMesh &mesh, const std::array<uint, 4> &tet, double det) {
    std::array<std::array<double, 3>, 4> gradients;
    dvec3 columns[2];
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 3; ++j) {
            uint ni = 0;
            for (uint ii = 0; ii < 4; ++ii) {
                if (ii == i) continue;
                uint nj = 0;
                for (uint jj = 0; jj < 3; ++jj) {
                    if (jj != j) columns[nj++][ni] = mesh.Points[tet[ii]][jj];
                }
                ++ni;
            }
            const int sign = (i + j) % 2 == 0 ? -1 : 1;
            gradients[i][j] = sign * numeric::Dot(dvec3{1}, numeric::Cross(columns[0], columns[1])) / det;
        }
    }
    return gradients;
}

template<typename Element>
void ElementStiffness(const Element &element, uint a, uint c, double lambda, double mu, modal::Tet10Assembler::ElementBlock &stiffness) {
    const auto &basis = ReferenceBasis();
    double g[3][3]{};
    for (uint k = 0; k < 4; ++k) {
        for (uint l = 0; l < 4; ++l) {
            const double weight = basis.Grad[a][k][c][l];
            if (weight == 0) continue;
            for (uint p = 0; p < 3; ++p) {
                for (uint q = 0; q < 3; ++q) g[p][q] += weight * element.Gradients[k][p] * element.Gradients[l][q];
            }
        }
    }
    const double trace = g[0][0] + g[1][1] + g[2][2];
    for (uint p = 0; p < 3; ++p) {
        for (uint q = 0; q < 3; ++q) stiffness[p + 3 * q] = element.Volume * (lambda * g[p][q] + mu * g[q][p] + (p == q ? mu * trace : 0));
    }
}

} // namespace

std::shared_ptr<const modal::Tet10Assembler::Topology> modal::Tet10Assembler::BuildTopology(const TetMesh &mesh) {
    auto state = std::make_shared<Topology>();
    state->NumNodes = uint32_t(mesh.Points.size());
    state->Elements.resize(mesh.Tets.size());
    std::unordered_map<uint64_t, uint> edge_nodes;
    edge_nodes.reserve(mesh.Tets.size() * 2);

    for (uint element_i = 0; element_i < state->Elements.size(); ++element_i) {
        auto &element = state->Elements[element_i];
        const auto &tet = mesh.Tets[element_i];
        for (uint corner = 0; corner < 4; ++corner) element.Nodes[corner] = tet[corner];
        for (uint edge = 0; edge < 6; ++edge) {
            const uint a = tet[EdgeCorners[edge][0]], b = tet[EdgeCorners[edge][1]];
            const uint64_t key = (uint64_t(std::min(a, b)) << 32) | std::max(a, b);
            const auto [entry, inserted] = edge_nodes.try_emplace(key, state->NumNodes);
            if (inserted) ++state->NumNodes;
            element.Nodes[4 + edge] = entry->second;
        }

        const dvec3 &a = mesh.Points[tet[0]], &b = mesh.Points[tet[1]], &c = mesh.Points[tet[2]], &d = mesh.Points[tet[3]];
        const double det = Determinant(a, b, c, d);
        element.Volume = std::abs(det / 6);
        element.Gradients = ShapeGradients(mesh, tet, det);
    }
    return state;
}

modal::Tet10Assembler::Tet10Assembler(const TetMesh &mesh, const AcousticMaterialProperties &material)
    : Tet10Assembler(BuildTopology(mesh), material) {}

modal::Tet10Assembler::Tet10Assembler(std::shared_ptr<const Topology> state, const AcousticMaterialProperties &material)
    : State(std::move(state)), NumNodes(State->NumNodes), Density(material.Density), Lambda(material.Lambda()), Mu(material.Mu()) {}

void modal::Tet10Assembler::EvaluateBlock(
    const Element &element, uint32_t row, uint32_t column, ElementBlock &stiffness, double &mass
) const {
    const auto &basis = ReferenceBasis();
    ElementStiffness(element, row, column, Lambda, Mu, stiffness);
    mass = Density * element.Volume * basis.Mass[row][column];
}

modal::AssembledPencil modal::Tet10Assembler::AssembleLower() const {
    std::vector<Eigen::Triplet<double>> mass_triplets, stiffness_triplets;
    mass_triplets.reserve(Elements().size() * LowerBlocksPerElement * 3);
    stiffness_triplets.reserve(Elements().size() * LowerBlocksPerElement * 9);

    for (const auto &element : Elements()) {
        for (uint a = 0; a < NodesPerElement; ++a) {
            for (uint c = 0; c <= a; ++c) {
                const bool transpose = element.Nodes[a] < element.Nodes[c];
                const uint local_row = transpose ? c : a, local_column = transpose ? a : c;
                const uint row = 3 * element.Nodes[local_row], column = 3 * element.Nodes[local_column];
                ElementBlock stiffness;
                double mass;
                EvaluateBlock(element, local_row, local_column, stiffness, mass);
                for (uint component = 0; component < 3; ++component) mass_triplets.emplace_back(row + component, column + component, mass);
                for (uint p = 0; p < 3; ++p) {
                    for (uint q = 0; q < (row == column ? p + 1 : 3u); ++q) stiffness_triplets.emplace_back(row + p, column + q, stiffness[p + 3 * q]);
                }
            }
        }
    }

    AssembledPencil assembled{Eigen::SparseMatrix<double>{Dofs(), Dofs()}, Eigen::SparseMatrix<double>{Dofs(), Dofs()}};
    assembled.Mass.setFromTriplets(mass_triplets.begin(), mass_triplets.end());
    assembled.Stiffness.setFromTriplets(stiffness_triplets.begin(), stiffness_triplets.end());
    return assembled;
}
