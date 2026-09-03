#include "FiniteCell.h"
#include "FiniteCellOracle.h"

#include <dispatch/dispatch.h>

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <Eigen/QR>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace {
using Clock = std::chrono::steady_clock;

constexpr std::array<std::array<int, 3>, 8> CornerSigns{{
    {-1, -1, -1},
    {1, -1, -1},
    {-1, 1, -1},
    {1, 1, -1},
    {-1, -1, 1},
    {1, -1, 1},
    {-1, 1, 1},
    {1, 1, 1},
}};
constexpr uint32_t BasisOrder{2}, BasisWidth{BasisOrder + 1}, NodeCount{modal::FiniteCellOperator::NodesPerCell}, LocalDofs{3 * NodeCount};
static_assert(NodeCount == BasisWidth * BasisWidth * BasisWidth);
constexpr double Gauss3{0.77459666924148337704};
constexpr std::array QuadraturePositions{-Gauss3, 0.0, Gauss3};
constexpr std::array QuadratureWeights{5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};

double SecondsSince(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double Component(const dvec3 &v, uint32_t axis) {
    return axis == 0 ? v.x : axis == 1 ? v.y :
                                         v.z;
}

std::pair<double, double> BoxDistanceBounds(const dvec3 &center, const dvec3 &half, const dvec3 &point) {
    const dvec3 distance = numeric::Abs(center - point);
    return {numeric::Length(numeric::Max(distance - half, dvec3{0})), numeric::Length(distance + half)};
}

std::pair<double, double> BoxRadialDistanceBounds(const dvec3 &center, const dvec3 &half, const dvec3 &axis) {
    const dvec3 distance = numeric::Abs(center - axis);
    return {
        std::hypot(std::max(distance.x - half.x, 0.0), std::max(distance.y - half.y, 0.0)),
        std::hypot(distance.x + half.x, distance.y + half.y),
    };
}

dvec3 CellCenter(const modal::FiniteCellOperator &operation, const modal::FiniteCellOperator::Cell &cell) {
    return 0.5 * (operation.Nodes[cell.Nodes[0]] + operation.Nodes[cell.Nodes[modal::FiniteCellOperator::NodesPerCell - 1]]);
}

double PointTriangleDistanceSquared(const dvec3 &p, const dvec3 &a, const dvec3 &b, const dvec3 &c) {
    const dvec3 ab = b - a, ac = c - a, ap = p - a;
    const double d1 = numeric::Dot(ab, ap), d2 = numeric::Dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return numeric::Length2(ap);

    const dvec3 bp = p - b;
    const double d3 = numeric::Dot(ab, bp), d4 = numeric::Dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return numeric::Length2(bp);
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        const dvec3 nearest = a + (d1 / (d1 - d3)) * ab;
        return numeric::Length2(p - nearest);
    }

    const dvec3 cp = p - c;
    const double d5 = numeric::Dot(ab, cp), d6 = numeric::Dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return numeric::Length2(cp);
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        const dvec3 nearest = a + (d2 / (d2 - d6)) * ac;
        return numeric::Length2(p - nearest);
    }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
        const dvec3 nearest = b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b);
        return numeric::Length2(p - nearest);
    }
    const dvec3 normal = numeric::Cross(ab, ac);
    const double projection = numeric::Dot(ap, normal);
    return projection * projection / numeric::Length2(normal);
}

bool RayHitsTriangle(const dvec3 &origin, const dvec3 &direction, const std::array<dvec3, 3> &triangle) {
    const dvec3 edge1 = triangle[1] - triangle[0], edge2 = triangle[2] - triangle[0];
    const dvec3 p = numeric::Cross(direction, edge2);
    const double det = numeric::Dot(edge1, p);
    const double scale = std::max({numeric::Length(edge1), numeric::Length(edge2), 1.0});
    if (std::abs(det) <= 1e-14 * scale * scale) return false;
    const double inverse = 1 / det;
    const dvec3 t = origin - triangle[0];
    const double u = numeric::Dot(t, p) * inverse;
    if (u < 0 || u > 1) return false;
    const dvec3 q = numeric::Cross(t, edge1);
    const double v = numeric::Dot(direction, q) * inverse;
    return v >= 0 && u + v <= 1 && numeric::Dot(edge2, q) * inverse > 1e-13 * scale;
}

bool TriangleIntersectsBox(const std::array<dvec3, 3> &triangle, const dvec3 &center, const dvec3 &half) {
    const std::array vertices{triangle[0] - center, triangle[1] - center, triangle[2] - center};
    const std::array edges{vertices[1] - vertices[0], vertices[2] - vertices[1], vertices[0] - vertices[2]};
    const auto Separates = [&](const dvec3 &axis) {
        if (numeric::Length2(axis) <= 1e-30) return false;
        const std::array projection{
            numeric::Dot(vertices[0], axis),
            numeric::Dot(vertices[1], axis),
            numeric::Dot(vertices[2], axis),
        };
        const double radius = numeric::Dot(half, numeric::Abs(axis));
        return std::ranges::min(projection) > radius || std::ranges::max(projection) < -radius;
    };
    for (uint32_t axis = 0; axis < 3; ++axis) {
        double min = Component(vertices[0], axis), max = min;
        for (uint32_t vertex = 1; vertex < 3; ++vertex) {
            min = std::min(min, Component(vertices[vertex], axis));
            max = std::max(max, Component(vertices[vertex], axis));
        }
        if (min > Component(half, axis) || max < -Component(half, axis)) return false;
    }
    if (Separates(numeric::Cross(edges[0], edges[1]))) return false;
    constexpr std::array<dvec3, 3> axes{dvec3{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (const auto &edge : edges)
        for (const auto &axis : axes)
            if (Separates(numeric::Cross(edge, axis))) return false;
    return true;
}

// Three non-axis-aligned ray directions provide independent parity classifications for a closed surface.
const std::array<dvec3, 3> InsideTestDirections{
    numeric::Normalize(dvec3{1, 0.3713906763541037, 0.6947465906068658}),
    numeric::Normalize(dvec3{0.217347184, 1, 0.513938221}),
    numeric::Normalize(dvec3{0.623741391, 0.281731491, 1}),
};

struct TriangleSurface {
    struct Node {
        dvec3 Min{std::numeric_limits<double>::infinity()}, Max{-std::numeric_limits<double>::infinity()};
        uint32_t Begin{}, Count{}, Left{}, Right{};
    };

    std::vector<std::array<dvec3, 3>> Triangles;
    std::vector<uint32_t> Indices;
    std::vector<Node> Nodes;
    static std::pair<dvec3, dvec3> TriangleBounds(const std::array<dvec3, 3> &triangle) {
        return {
            numeric::Min(triangle[0], numeric::Min(triangle[1], triangle[2])),
            numeric::Max(triangle[0], numeric::Max(triangle[1], triangle[2])),
        };
    }

    uint32_t BuildNode(uint32_t begin, uint32_t count) {
        const uint32_t node_index = uint32_t(Nodes.size());
        Nodes.emplace_back();
        Node node{.Begin = begin, .Count = count};
        dvec3 centroid_min{std::numeric_limits<double>::infinity()}, centroid_max{-std::numeric_limits<double>::infinity()};
        for (uint32_t entry = begin; entry < begin + count; ++entry) {
            const auto [min, max] = TriangleBounds(Triangles[Indices[entry]]);
            node.Min = numeric::Min(node.Min, min);
            node.Max = numeric::Max(node.Max, max);
            centroid_min = numeric::Min(centroid_min, 0.5 * (min + max));
            centroid_max = numeric::Max(centroid_max, 0.5 * (min + max));
        }
        if (count > 8) {
            const dvec3 extent = centroid_max - centroid_min;
            const uint32_t axis = extent.x >= extent.y && extent.x >= extent.z ? 0 : extent.y >= extent.z ? 1 :
                                                                                                            2;
            std::sort(Indices.begin() + begin, Indices.begin() + begin + count, [&](uint32_t a, uint32_t b) {
                const auto [a_min, a_max] = TriangleBounds(Triangles[a]);
                const auto [b_min, b_max] = TriangleBounds(Triangles[b]);
                const double ca = Component(a_min + a_max, axis), cb = Component(b_min + b_max, axis);
                return ca != cb ? ca < cb : a < b;
            });
            const uint32_t left_count = count / 2;
            node.Left = BuildNode(begin, left_count);
            node.Right = BuildNode(begin + left_count, count - left_count);
            node.Count = 0;
        }
        Nodes[node_index] = node;
        return node_index;
    }

    void Build() {
        Indices.resize(Triangles.size());
        std::iota(Indices.begin(), Indices.end(), 0);
        Nodes.reserve(2 * Triangles.size());
        BuildNode(0, uint32_t(Triangles.size()));
    }

    static double BoxDistanceSquared(const Node &node, const dvec3 &point) {
        const dvec3 outside = numeric::Max(numeric::Max(node.Min - point, point - node.Max), dvec3{0});
        return numeric::Length2(outside);
    }

    static bool RayIntersectsBox(const Node &node, const dvec3 &origin, const dvec3 &direction) {
        double near = -std::numeric_limits<double>::infinity(), far = std::numeric_limits<double>::infinity();
        for (uint32_t axis = 0; axis < 3; ++axis) {
            const double d = Component(direction, axis), o = Component(origin, axis);
            const double min = Component(node.Min, axis), max = Component(node.Max, axis);
            double a = (min - o) / d, b = (max - o) / d;
            if (a > b) std::swap(a, b);
            near = std::max(near, a);
            far = std::min(far, b);
        }
        return far >= std::max(near, 0.0);
    }

    uint32_t RayIntersections(const dvec3 &point, const dvec3 &direction) const {
        std::array<uint32_t, 64> stack{};
        uint32_t size{1}, intersections{};
        while (size) {
            const Node &node = Nodes[stack[--size]];
            if (!RayIntersectsBox(node, point, direction)) continue;
            if (node.Count) {
                for (uint32_t entry = node.Begin; entry < node.Begin + node.Count; ++entry)
                    intersections += RayHitsTriangle(point, direction, Triangles[Indices[entry]]);
            } else {
                stack[size++] = node.Left;
                stack[size++] = node.Right;
            }
        }
        return intersections;
    }

    static bool BoxesOverlap(const Node &node, const dvec3 &center, const dvec3 &half) {
        const dvec3 min = center - half, max = center + half;
        return node.Min.x <= max.x && node.Min.y <= max.y && node.Min.z <= max.z && node.Max.x >= min.x && node.Max.y >= min.y && node.Max.z >= min.z;
    }

    double SignedDistance(const dvec3 &point) const {
        double squared = std::numeric_limits<double>::infinity();
        std::array<uint32_t, 64> stack{};
        uint32_t size{1};
        while (size) {
            const Node &node = Nodes[stack[--size]];
            if (BoxDistanceSquared(node, point) >= squared) continue;
            if (node.Count) {
                for (uint32_t entry = node.Begin; entry < node.Begin + node.Count; ++entry) {
                    const auto &triangle = Triangles[Indices[entry]];
                    squared = std::min(squared, PointTriangleDistanceSquared(point, triangle[0], triangle[1], triangle[2]));
                }
                continue;
            }
            const double left = BoxDistanceSquared(Nodes[node.Left], point), right = BoxDistanceSquared(Nodes[node.Right], point);
            if (left < right) {
                stack[size++] = node.Right;
                stack[size++] = node.Left;
            } else {
                stack[size++] = node.Left;
                stack[size++] = node.Right;
            }
        }
        if (squared <= 1e-28) return 0;

        uint32_t inside_votes{};
        for (const auto &direction : InsideTestDirections) inside_votes += RayIntersections(point, direction) % 2;
        const double distance = std::sqrt(squared);
        return inside_votes >= 2 ? -distance : distance;
    }

    modal::DomainRegion ClassifyBox(const dvec3 &center, const dvec3 &half) const {
        std::array<uint32_t, 64> stack{};
        uint32_t size{1};
        while (size) {
            const Node &node = Nodes[stack[--size]];
            if (!BoxesOverlap(node, center, half)) continue;
            if (node.Count) {
                for (uint32_t entry = node.Begin; entry < node.Begin + node.Count; ++entry)
                    if (TriangleIntersectsBox(Triangles[Indices[entry]], center, half)) return modal::DomainRegion::Cut;
            } else {
                stack[size++] = node.Left;
                stack[size++] = node.Right;
            }
        }
        return SignedDistance(center) < 0 ? modal::DomainRegion::Inside : modal::DomainRegion::Outside;
    }
};

modal::DomainRegion Classify(const modal::ImplicitDomain &domain, const dvec3 &center, const dvec3 &half) {
    if (domain.ClassifyBox) return domain.ClassifyBox(center, half);
    const double distance = domain.SignedDistance(center), radius = numeric::Length(half);
    if (distance < -radius) return modal::DomainRegion::Inside;
    if (distance > radius) return modal::DomainRegion::Outside;
    return modal::DomainRegion::Cut;
}

template<uint32_t Order>
void EvaluateBasis(
    const dvec3 &reference, const dvec3 &inverse_half,
    std::array<double, (Order + 1) * (Order + 1) * (Order + 1)> &shape,
    std::array<dvec3, (Order + 1) * (Order + 1) * (Order + 1)> &gradient
) {
    static constexpr uint32_t Width{Order + 1};
    std::array<std::array<double, Width>, 3> basis{}, derivative{};
    for (uint32_t axis = 0; axis < 3; ++axis) {
        const double x = Component(reference, axis);
        if constexpr (Order == 1) {
            basis[axis] = {0.5 * (1 - x), 0.5 * (1 + x)};
            derivative[axis] = {-0.5, 0.5};
        } else {
            basis[axis] = {0.5 * x * (x - 1), 1 - x * x, 0.5 * x * (x + 1)};
            derivative[axis] = {x - 0.5, -2 * x, x + 0.5};
        }
    }
    for (uint32_t z = 0; z < Width; ++z) {
        for (uint32_t y = 0; y < Width; ++y) {
            for (uint32_t x = 0; x < Width; ++x) {
                const uint32_t node = x + Width * (y + Width * z);
                shape[node] = basis[0][x] * basis[1][y] * basis[2][z];
                gradient[node] = {
                    derivative[0][x] * basis[1][y] * basis[2][z] * inverse_half.x,
                    basis[0][x] * derivative[1][y] * basis[2][z] * inverse_half.y,
                    basis[0][x] * basis[1][y] * derivative[2][z] * inverse_half.z,
                };
            }
        }
    }
}

struct CellIntegrator {
    const modal::ImplicitDomain &Domain;
    const modal::FiniteCellConfig &Config;
    dvec3 CellCenter, CellHalf;
    std::vector<modal::FiniteCellOperator::QuadraturePoint> &Quadrature;
    double PhysicalVolume{};

    void Sample(const dvec3 &point, double weight, double coefficient) {
        if (coefficient == 1) PhysicalVolume += weight;
        Quadrature.push_back({(point - CellCenter) / CellHalf, weight * coefficient, coefficient != 1});
    }

    void IntegrateQuadrature(const dvec3 &center, const dvec3 &half, double coefficient, bool classify_points) {
        const double jacobian = half.x * half.y * half.z;
        for (uint32_t z = 0; z < BasisWidth; ++z) {
            for (uint32_t y = 0; y < BasisWidth; ++y) {
                for (uint32_t x = 0; x < BasisWidth; ++x) {
                    const dvec3 point = center + dvec3{QuadraturePositions[x] * half.x, QuadraturePositions[y] * half.y, QuadraturePositions[z] * half.z};
                    const double point_coefficient = classify_points ?
                        (Domain.SignedDistance(point) <= 0 ? 1 : Config.FictitiousScale) :
                        coefficient;
                    Sample(point, jacobian * QuadratureWeights[x] * QuadratureWeights[y] * QuadratureWeights[z], point_coefficient);
                }
            }
        }
    }

    void IntegrateChildren(const dvec3 &center, const dvec3 &half, uint32_t depth) {
        const dvec3 child_half = 0.5 * half;
        for (const auto &sign : CornerSigns)
            Integrate(center + dvec3{sign[0] * child_half.x, sign[1] * child_half.y, sign[2] * child_half.z}, child_half, depth + 1);
    }

    void Integrate(const dvec3 &center, const dvec3 &half, uint32_t depth) {
        switch (Classify(Domain, center, half)) {
            case modal::DomainRegion::Inside: IntegrateQuadrature(center, half, 1, false); return;
            case modal::DomainRegion::Outside: IntegrateQuadrature(center, half, Config.FictitiousScale, false); return;
            case modal::DomainRegion::Cut: break;
        }
        if (depth == Config.CutDepth) {
            IntegrateQuadrature(center, half, 0, true);
            return;
        }
        IntegrateChildren(center, half, depth);
    }
};

constexpr uint32_t MomentDegree{2 * BasisOrder}, MomentWidth{MomentDegree + 1}, MomentCount{MomentWidth * MomentWidth * MomentWidth};
using MomentVector = Eigen::Matrix<double, MomentCount, 1>;

MomentVector MomentBasis(const dvec3 &point) {
    std::array<std::array<double, MomentWidth>, 3> legendre{};
    for (uint32_t axis = 0; axis < 3; ++axis) {
        const double x = Component(point, axis);
        legendre[axis][0] = 1;
        legendre[axis][1] = x;
        for (uint32_t degree = 2; degree <= MomentDegree; ++degree)
            legendre[axis][degree] = ((2 * degree - 1) * x * legendre[axis][degree - 1] -
                                      (degree - 1) * legendre[axis][degree - 2]) /
                degree;
    }
    MomentVector result;
    for (uint32_t z = 0; z < MomentWidth; ++z)
        for (uint32_t y = 0; y < MomentWidth; ++y)
            for (uint32_t x = 0; x < MomentWidth; ++x)
                result[x + MomentWidth * (y + MomentWidth * z)] = legendre[0][x] * legendre[1][y] * legendre[2][z];
    return result;
}

struct MomentFitResult {
    std::vector<modal::FiniteCellOperator::QuadraturePoint> Points;
    double Residual{};
};

std::optional<MomentFitResult> FitCutCellMoments(
    std::span<const modal::FiniteCellOperator::QuadraturePoint> oracle,
    std::span<const modal::FiniteCellOperator::QuadraturePoint> candidates
) {
    if (oracle.size() % NodeCount) throw std::logic_error("Finite-cell oracle quadrature has an incomplete tensor patch.");
    if (candidates.size() % NodeCount) throw std::logic_error("Finite-cell candidate quadrature has an incomplete tensor patch.");
    if (candidates.size() >= oracle.size()) return std::nullopt;

    std::array<double, 2> measures{};
    std::array<MomentVector, 2> moments{};
    for (auto &moment : moments) moment.setZero();
    for (const auto &sample : oracle) {
        const uint32_t region = sample.Fictitious;
        measures[region] += sample.Weight;
        moments[region] += sample.Weight * MomentBasis(sample.Reference);
    }
    for (uint32_t region = 0; region < 2; ++region)
        if (measures[region] > 0) moments[region] /= measures[region];
    std::array<std::vector<double>, 2> candidate_weights{
        std::vector<double>(candidates.size()),
        std::vector<double>(candidates.size()),
    };
    std::vector<double> oracle_weights(oracle.size());
    double maximum_residual{};
    Eigen::MatrixXd basis;
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> decomposition;
    if (candidates.size() >= MomentCount && (measures[0] > 0 || measures[1] > 0)) {
        basis.resize(MomentCount, candidates.size());
        for (uint32_t candidate = 0; candidate < candidates.size(); ++candidate)
            basis.col(candidate) = MomentBasis(candidates[candidate].Reference);
        decomposition.compute(basis);
    }
    for (uint32_t region = 0; region < 2; ++region) {
        bool fitted_region{};
        if (basis.size() && measures[region] > 0) {
            const Eigen::VectorXd fitted = decomposition.solve(moments[region]);
            const double residual = (basis * fitted - moments[region]).norm() / moments[region].norm();
            fitted_region = fitted.allFinite() && residual <= 1e-10;
            if (fitted_region) {
                maximum_residual = std::max(maximum_residual, residual);
                for (uint32_t candidate = 0; candidate < candidates.size(); ++candidate)
                    candidate_weights[region][candidate] = measures[region] * fitted[candidate];
            }
        }
        if (!fitted_region)
            for (uint32_t index = 0; index < oracle.size(); ++index)
                if (oracle[index].Fictitious == region) oracle_weights[index] = oracle[index].Weight;
    }
    std::vector<modal::FiniteCellOperator::QuadraturePoint> points;
    const auto Append = [&](const auto &source, const std::vector<double> &weights, std::optional<uint8_t> region = std::nullopt) {
        const uint32_t patch_count = uint32_t(source.size()) / NodeCount;
        for (uint32_t patch = 0; patch < patch_count; ++patch) {
            bool retained{};
            for (uint32_t point = 0; point < NodeCount; ++point)
                retained |= weights[patch * NodeCount + point] != 0;
            if (!retained) continue;
            for (uint32_t point = 0; point < NodeCount; ++point) {
                const uint32_t index = patch * NodeCount + point;
                auto fitted = source[index];
                fitted.Weight = weights[index];
                if (region.has_value()) fitted.Fictitious = *region;
                points.push_back(fitted);
            }
        }
    };
    Append(candidates, candidate_weights[0], uint8_t{0});
    Append(candidates, candidate_weights[1], uint8_t{1});
    Append(oracle, oracle_weights);
    if (points.size() < oracle.size()) return MomentFitResult{std::move(points), maximum_residual};
    return std::nullopt;
}

void BuildMomentFittedQuadrature(
    modal::FiniteCellOperator &operation, const modal::ImplicitDomain &domain,
    const modal::FiniteCellConfig &config
) {
    struct CellQuadrature {
        std::vector<modal::FiniteCellOperator::QuadraturePoint> Points;
        double PhysicalVolume{}, Residual{};
        uint32_t OracleCount{};
        bool Fitted{};
    };
    struct Context {
        const modal::FiniteCellOperator &Operation;
        const modal::ImplicitDomain &Domain;
        const modal::FiniteCellConfig &Config;
        std::vector<CellQuadrature> &Quadrature;
    };
    std::vector<CellQuadrature> cells(operation.Cells.size());
    Context context{operation, domain, config, cells};
    dispatch_apply_f(
        operation.Cells.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &context,
        [](void *raw, size_t cell_index) {
            auto &context = *static_cast<Context *>(raw);
            const auto &cell = context.Operation.Cells[cell_index];
            auto &result = context.Quadrature[cell_index];
            const dvec3 center = CellCenter(context.Operation, cell), half = 1.0 / cell.InverseHalf;
            std::vector<modal::FiniteCellOperator::QuadraturePoint> oracle;
            CellIntegrator oracle_integrator{context.Domain, context.Config, center, half, oracle};
            oracle_integrator.Integrate(center, half, 0);
            result.OracleCount = uint32_t(oracle.size());
            result.PhysicalVolume = oracle_integrator.PhysicalVolume;
            if (cell.Cut) {
                for (uint32_t depth = 0; depth < context.Config.CutDepth; ++depth) {
                    auto candidate_config = context.Config;
                    candidate_config.CutDepth = depth;
                    std::vector<modal::FiniteCellOperator::QuadraturePoint> candidates;
                    CellIntegrator candidate_integrator{context.Domain, candidate_config, center, half, candidates};
                    candidate_integrator.Integrate(center, half, 0);
                    if (auto fit = FitCutCellMoments(oracle, candidates)) {
                        result.Points = std::move(fit->Points);
                        result.Residual = fit->Residual;
                        result.Fitted = true;
                        return;
                    }
                }
            }
            result.Points = std::move(oracle);
        }
    );

    size_t point_count{};
    for (const auto &cell : cells) point_count += cell.Points.size();
    operation.Quadrature.resize(point_count);
    uint32_t write{};
    for (uint32_t cell_index = 0; cell_index < operation.Cells.size(); ++cell_index) {
        auto &cell = operation.Cells[cell_index];
        auto &quadrature = cells[cell_index];
        cell.QuadratureOffset = write;
        cell.QuadratureCount = uint32_t(quadrature.Points.size());
        cell.OracleQuadratureCount = quadrature.OracleCount;
        std::move(quadrature.Points.begin(), quadrature.Points.end(), operation.Quadrature.begin() + write);
        write += cell.QuadratureCount;
        operation.Profile.PhysicalVolume += quadrature.PhysicalVolume;
        operation.Profile.OracleQuadraturePoints += quadrature.OracleCount;
        operation.Profile.QuadraturePoints += cell.QuadratureCount;
        if (cell.Cut) {
            if (quadrature.Fitted) {
                cell.MomentFitted = true;
                operation.Profile.MomentFitMaximumResidual = std::max(operation.Profile.MomentFitMaximumResidual, quadrature.Residual);
                ++operation.Profile.MomentFittedCells;
            } else {
                cell.MomentFitFallback = true;
                ++operation.Profile.MomentFitFallbackCells;
            }
        }
    }
}

uint32_t BackgroundNode(uvec3 point, uvec3 dimensions) {
    return (point.x * dimensions.y + point.y) * dimensions.z + point.z;
}

void BuildP1Transfer(modal::FiniteCellOperator &operation) {
    std::vector<int32_t> coarse(operation.Nodes.size(), -1);
    for (const auto &cell : operation.Cells) {
        for (uint32_t z = 0; z < 2; ++z) {
            for (uint32_t y = 0; y < 2; ++y) {
                for (uint32_t x = 0; x < 2; ++x) {
                    const uint32_t local = BasisOrder * x + BasisWidth * (BasisOrder * y + BasisWidth * BasisOrder * z);
                    int32_t &index = coarse[cell.Nodes[local]];
                    if (index < 0) index = int32_t(operation.NumP1Nodes++);
                }
            }
        }
    }

    operation.P1Stencils.resize(operation.Nodes.size());
    for (const auto &cell : operation.Cells) {
        std::array<uint32_t, 8> corners{};
        for (uint32_t z = 0; z < 2; ++z)
            for (uint32_t y = 0; y < 2; ++y)
                for (uint32_t x = 0; x < 2; ++x) {
                    const uint32_t corner = x + 2 * (y + 2 * z);
                    const uint32_t local = BasisOrder * x + BasisWidth * (BasisOrder * y + BasisWidth * BasisOrder * z);
                    corners[corner] = uint32_t(coarse[cell.Nodes[local]]);
                }

        for (uint32_t z = 0; z < BasisWidth; ++z) {
            for (uint32_t y = 0; y < BasisWidth; ++y) {
                for (uint32_t x = 0; x < BasisWidth; ++x) {
                    const uint32_t local = x + BasisWidth * (y + BasisWidth * z), fine = cell.Nodes[local];
                    auto &stencil = operation.P1Stencils[fine];
                    if (stencil.Count) continue;
                    const std::array<double, 2> wx{1 - double(x) / BasisOrder, double(x) / BasisOrder};
                    const std::array<double, 2> wy{1 - double(y) / BasisOrder, double(y) / BasisOrder};
                    const std::array<double, 2> wz{1 - double(z) / BasisOrder, double(z) / BasisOrder};
                    for (uint32_t cz = 0; cz < 2; ++cz)
                        for (uint32_t cy = 0; cy < 2; ++cy)
                            for (uint32_t cx = 0; cx < 2; ++cx) {
                                const double weight = wx[cx] * wy[cy] * wz[cz];
                                if (weight == 0) continue;
                                const uint32_t entry = stencil.Count++;
                                stencil.Nodes[entry] = corners[cx + 2 * (cy + 2 * cz)];
                                stencil.Weights[entry] = weight;
                            }
                }
            }
        }
    }
}

modal::FiniteCellOperator Build(
    const modal::ImplicitDomain &domain, const AcousticMaterialProperties &material,
    const modal::FiniteCellConfig &config, Clock::time_point start, bool moment_fitting
) {
    modal::FiniteCellOperator result;
    result.Density = material.Density;
    result.Lambda = material.Lambda();
    result.Mu = material.Mu();
    result.FictitiousScale = config.FictitiousScale;
    const dvec3 physical_extent = domain.Max - domain.Min;
    const dvec3 nominal_step = physical_extent / dvec3{config.Cells};
    const dvec3 grid_min = domain.Min + (config.GridOffsetCells - config.PaddingCells) * nominal_step;
    const dvec3 grid_max = domain.Max + (config.GridOffsetCells + config.PaddingCells) * nominal_step;
    const dvec3 step = (grid_max - grid_min) / dvec3{config.Cells};
    result.GridCells = config.Cells;
    result.GridMin = grid_min;
    result.CellStep = step;
    const dvec3 node_step = step / double(BasisOrder), half = 0.5 * step;
    const uvec3 node_dimensions = BasisOrder * config.Cells + uvec3{1};
    const size_t background_nodes = size_t(node_dimensions.x) * node_dimensions.y * node_dimensions.z;
    std::vector<int32_t> compact(background_nodes, -1);
    result.Profile.BackgroundCells = config.Cells.x * config.Cells.y * config.Cells.z;
    result.CellAtBackgroundIndex.resize(result.Profile.BackgroundCells, -1);

    const auto acquire_node = [&](uvec3 point) {
        const uint32_t background = BackgroundNode(point, node_dimensions);
        int32_t &node = compact[background];
        if (node < 0) {
            node = int32_t(result.Nodes.size());
            result.Nodes.push_back(grid_min + dvec3{point} * node_step);
        }
        return uint32_t(node);
    };

    for (uint32_t x = 0; x < config.Cells.x; ++x) {
        for (uint32_t y = 0; y < config.Cells.y; ++y) {
            for (uint32_t z = 0; z < config.Cells.z; ++z) {
                const dvec3 center = grid_min + (dvec3{double(x), double(y), double(z)} + 0.5) * step;
                const modal::DomainRegion region = Classify(domain, center, half);
                if (region == modal::DomainRegion::Outside) continue;
                ++result.Profile.ActiveCells;
                result.Profile.CutCells += region == modal::DomainRegion::Cut;
                modal::FiniteCellOperator::Cell cell;
                cell.InverseHalf = 1.0 / half;
                cell.QuadratureOffset = uint32_t(result.Quadrature.size());
                cell.Color = uint8_t((x & 1) | ((y & 1) << 1) | ((z & 1) << 2));
                cell.Cut = region == modal::DomainRegion::Cut;
                if (!moment_fitting) {
                    CellIntegrator integrator{domain, config, center, half, result.Quadrature};
                    integrator.Integrate(center, half, 0);
                    cell.OracleQuadratureCount = uint32_t(result.Quadrature.size()) - cell.QuadratureOffset;
                    cell.QuadratureCount = cell.OracleQuadratureCount;
                    result.Profile.PhysicalVolume += integrator.PhysicalVolume;
                    result.Profile.OracleQuadraturePoints += cell.OracleQuadratureCount;
                    result.Profile.QuadraturePoints += cell.QuadratureCount;
                }

                for (uint32_t local_z = 0; local_z < BasisWidth; ++local_z) {
                    for (uint32_t local_y = 0; local_y < BasisWidth; ++local_y) {
                        for (uint32_t local_x = 0; local_x < BasisWidth; ++local_x) {
                            const uint32_t node = local_x + BasisWidth * (local_y + BasisWidth * local_z);
                            cell.Nodes[node] = acquire_node({BasisOrder * x + local_x, BasisOrder * y + local_y, BasisOrder * z + local_z});
                        }
                    }
                }
                result.CellAtBackgroundIndex[(x * config.Cells.y + y) * config.Cells.z + z] = int32_t(result.Cells.size());
                result.Cells.push_back(cell);
            }
        }
    }
    if (moment_fitting) BuildMomentFittedQuadrature(result, domain, config);
    BuildP1Transfer(result);
    static_assert(NodeCount <= 32);
    result.NodeOccurrenceOffsets.resize(result.Nodes.size() + 1);
    for (const auto &cell : result.Cells)
        for (uint32_t local = 0; local < NodeCount; ++local) ++result.NodeOccurrenceOffsets[cell.Nodes[local] + 1];
    std::partial_sum(
        result.NodeOccurrenceOffsets.begin(), result.NodeOccurrenceOffsets.end(),
        result.NodeOccurrenceOffsets.begin()
    );
    result.NodeOccurrences.resize(result.NodeOccurrenceOffsets.back());
    std::vector<uint32_t> occurrence_cursor = result.NodeOccurrenceOffsets;
    for (uint32_t cell = 0; cell < result.Cells.size(); ++cell)
        for (uint32_t local = 0; local < NodeCount; ++local) {
            const uint32_t node = result.Cells[cell].Nodes[local];
            result.NodeOccurrences[occurrence_cursor[node]++] = cell << 5 | local;
        }
    result.Profile.Dofs = 3 * uint32_t(result.Nodes.size());
    result.Profile.Assemble = SecondsSince(start);
    return result;
}

using Tensor = std::array<double, NodeCount>;
using TensorMatrix = std::array<std::array<double, BasisWidth>, BasisWidth>;

void EvaluateBasis1D(double coordinate, double inverse_half, std::array<double, BasisWidth> &basis, std::array<double, BasisWidth> &derivative) {
    basis = {0.5 * coordinate * (coordinate - 1), 1 - coordinate * coordinate, 0.5 * coordinate * (coordinate + 1)};
    derivative = {
        (coordinate - 0.5) * inverse_half,
        -2 * coordinate * inverse_half,
        (coordinate + 0.5) * inverse_half,
    };
}

TensorMatrix Transpose(const TensorMatrix &matrix) {
    TensorMatrix result;
    for (uint32_t row = 0; row < BasisWidth; ++row)
        for (uint32_t column = 0; column < BasisWidth; ++column) result[row][column] = matrix[column][row];
    return result;
}

using Double2 = double __attribute__((ext_vector_type(2)));

Tensor TensorProduct(
    const Tensor &input, const TensorMatrix &x_matrix,
    const TensorMatrix &y_matrix, const TensorMatrix &z_matrix
) {
    Tensor x_values, xy_values, result;
    for (uint32_t z = 0; z < BasisWidth; ++z) {
        for (uint32_t y = 0; y < BasisWidth; ++y) {
            Double2 pair{};
            double last{};
            for (uint32_t x = 0; x < BasisWidth; ++x) {
                const double value = input[x + BasisWidth * (y + BasisWidth * z)];
                pair += Double2{x_matrix[0][x], x_matrix[1][x]} * value;
                last += x_matrix[2][x] * value;
            }
            const uint32_t offset = BasisWidth * (y + BasisWidth * z);
            x_values[offset] = pair[0];
            x_values[offset + 1] = pair[1];
            x_values[offset + 2] = last;
        }
    }
    for (uint32_t z = 0; z < BasisWidth; ++z) {
        for (uint32_t out_y = 0; out_y < BasisWidth; ++out_y) {
            Double2 pair{};
            double last{};
            for (uint32_t y = 0; y < BasisWidth; ++y) {
                const uint32_t offset = BasisWidth * (y + BasisWidth * z);
                const double coefficient = y_matrix[out_y][y];
                pair += Double2{x_values[offset], x_values[offset + 1]} * coefficient;
                last += x_values[offset + 2] * coefficient;
            }
            const uint32_t offset = BasisWidth * (out_y + BasisWidth * z);
            xy_values[offset] = pair[0];
            xy_values[offset + 1] = pair[1];
            xy_values[offset + 2] = last;
        }
    }
    for (uint32_t out_z = 0; out_z < BasisWidth; ++out_z) {
        for (uint32_t y = 0; y < BasisWidth; ++y) {
            Double2 pair{};
            double last{};
            for (uint32_t z = 0; z < BasisWidth; ++z) {
                const uint32_t offset = BasisWidth * (y + BasisWidth * z);
                const double coefficient = z_matrix[out_z][z];
                pair += Double2{xy_values[offset], xy_values[offset + 1]} * coefficient;
                last += xy_values[offset + 2] * coefficient;
            }
            const uint32_t offset = BasisWidth * (y + BasisWidth * out_z);
            result[offset] = pair[0];
            result[offset + 1] = pair[1];
            result[offset + 2] = last;
        }
    }
    return result;
}

// Evaluates `stiffness_scale*K + mass_scale*M` and an optional independent M action in one cell traversal.
// Cell destinations receive local actions, while global destinations receive accumulated nodal actions.
// Null destinations omit their corresponding actions.
void ApplyTensorSerial(
    const modal::FiniteCellOperator &operation, const double *input, double *output,
    uint32_t width, double stiffness_scale, double mass_scale, double *independent_mass_output = nullptr,
    uint32_t first_cell = 0, uint32_t cell_count = std::numeric_limits<uint32_t>::max(),
    double *cell_output = nullptr, double *cell_mass_output = nullptr,
    const uint32_t *cell_indices = nullptr
) {
    const uint32_t dofs = operation.Dofs();
    const bool action = output || cell_output;
    const bool independent_mass = independent_mass_output || cell_mass_output;
    const bool needs_displacement = independent_mass || (action && mass_scale != 0);
    if (output) std::fill_n(output, size_t(dofs) * width, 0.0);
    if (independent_mass_output) std::fill_n(independent_mass_output, size_t(dofs) * width, 0.0);
    const size_t local_values = size_t(LocalDofs) * width;
    std::vector<double> local_storage, local_mass_storage;
    if (action && !cell_output) local_storage.resize(local_values);
    if (independent_mass && !cell_mass_output) local_mass_storage.resize(local_values);
    const uint32_t end_cell = cell_indices ? first_cell + cell_count :
                                             uint32_t(std::min<size_t>(operation.Cells.size(), size_t(first_cell) + cell_count));
    for (uint32_t ordinal = first_cell; ordinal < end_cell; ++ordinal) {
        const uint32_t cell_index = cell_indices ? cell_indices[ordinal] : ordinal;
        const auto &cell = operation.Cells[cell_index];
        const size_t cell_offset = size_t(cell_index) * local_values;
        double *local = cell_output ? cell_output + cell_offset : local_storage.data();
        double *local_mass = cell_mass_output ? cell_mass_output + cell_offset : local_mass_storage.data();
        if (action) std::fill_n(local, local_values, 0.0);
        if (independent_mass) std::fill_n(local_mass, local_values, 0.0);
        for (uint32_t patch = cell.QuadratureOffset; patch < cell.QuadratureOffset + cell.QuadratureCount; patch += NodeCount) {
            std::array<TensorMatrix, 3> basis{}, derivative{}, basis_transpose{}, derivative_transpose{};
            for (uint32_t axis = 0; axis < 3; ++axis) {
                const uint32_t stride = axis == 0 ? 1 : axis == 1 ? BasisWidth :
                                                                    BasisWidth * BasisWidth;
                for (uint32_t point = 0; point < BasisWidth; ++point)
                    EvaluateBasis1D(
                        Component(operation.Quadrature[patch + point * stride].Reference, axis),
                        Component(cell.InverseHalf, axis), basis[axis][point], derivative[axis][point]
                    );
                basis_transpose[axis] = Transpose(basis[axis]);
                derivative_transpose[axis] = Transpose(derivative[axis]);
            }

            for (uint32_t block = 0; block < width; ++block) {
                const size_t input_offset = size_t(block) * dofs;
                std::array<Tensor, 3> nodal, displacement, mass, mass_only;
                std::array<std::array<Tensor, 3>, 3> gradient, stress;
                for (uint32_t node = 0; node < NodeCount; ++node)
                    for (uint32_t component = 0; component < 3; ++component)
                        nodal[component][node] = input[input_offset + 3 * cell.Nodes[node] + component];
                for (uint32_t component = 0; component < 3; ++component) {
                    if (needs_displacement)
                        displacement[component] = TensorProduct(nodal[component], basis[0], basis[1], basis[2]);
                    if (action && stiffness_scale != 0) {
                        gradient[component][0] = TensorProduct(nodal[component], derivative[0], basis[1], basis[2]);
                        gradient[component][1] = TensorProduct(nodal[component], basis[0], derivative[1], basis[2]);
                        gradient[component][2] = TensorProduct(nodal[component], basis[0], basis[1], derivative[2]);
                    }
                }
                for (uint32_t point = 0; point < NodeCount; ++point) {
                    const double weight = operation.Quadrature[patch + point].Weight;
                    if (independent_mass)
                        for (uint32_t p = 0; p < 3; ++p) mass_only[p][point] = weight * operation.Density * displacement[p][point];
                    if (!action) continue;
                    if (mass_scale != 0)
                        for (uint32_t p = 0; p < 3; ++p) mass[p][point] = weight * mass_scale * operation.Density * displacement[p][point];
                    if (stiffness_scale == 0) continue;
                    const double trace = gradient[0][0][point] + gradient[1][1][point] + gradient[2][2][point];
                    for (uint32_t p = 0; p < 3; ++p)
                        for (uint32_t q = 0; q < 3; ++q)
                            stress[p][q][point] = weight * stiffness_scale *
                                ((p == q ? operation.Lambda * trace : 0) +
                                 operation.Mu * (gradient[p][q][point] + gradient[q][p][point]));
                }
                for (uint32_t component = 0; component < 3; ++component) {
                    if (action) {
                        Tensor result{};
                        if (mass_scale != 0)
                            result = TensorProduct(
                                mass[component], basis_transpose[0], basis_transpose[1], basis_transpose[2]
                            );
                        if (stiffness_scale != 0) {
                            const auto x_result = TensorProduct(
                                stress[component][0], derivative_transpose[0], basis_transpose[1], basis_transpose[2]
                            );
                            const auto y_result = TensorProduct(
                                stress[component][1], basis_transpose[0], derivative_transpose[1], basis_transpose[2]
                            );
                            const auto z_result = TensorProduct(
                                stress[component][2], basis_transpose[0], basis_transpose[1], derivative_transpose[2]
                            );
                            for (uint32_t node = 0; node < NodeCount; ++node) result[node] += x_result[node] + y_result[node] + z_result[node];
                        }
                        for (uint32_t node = 0; node < NodeCount; ++node)
                            local[size_t(block) * LocalDofs + 3 * node + component] += result[node];
                    }
                    if (independent_mass) {
                        const auto mass_result = TensorProduct(
                            mass_only[component], basis_transpose[0], basis_transpose[1], basis_transpose[2]
                        );
                        for (uint32_t node = 0; node < NodeCount; ++node)
                            local_mass[size_t(block) * LocalDofs + 3 * node + component] += mass_result[node];
                    }
                }
            }
        }
        if (!output && !independent_mass_output) continue;
        for (uint32_t block = 0; block < width; ++block) {
            const size_t output_offset = size_t(block) * dofs;
            if (output)
                for (uint32_t node = 0; node < NodeCount; ++node)
                    for (uint32_t component = 0; component < 3; ++component)
                        output[output_offset + 3 * cell.Nodes[node] + component] +=
                            local[size_t(block) * LocalDofs + 3 * node + component];
            if (independent_mass_output)
                for (uint32_t node = 0; node < NodeCount; ++node)
                    for (uint32_t component = 0; component < 3; ++component)
                        independent_mass_output[output_offset + 3 * cell.Nodes[node] + component] +=
                            local_mass[size_t(block) * LocalDofs + 3 * node + component];
        }
    }
}

struct CellGatherContext {
    const modal::FiniteCellOperator &Operation;
    const double *CellOutput, *CellMassOutput;
    double *Output, *MassOutput;
    uint32_t Width, NodesPerTask;
};

void ApplyTensorCellGatherTile(void *raw_context, size_t task) {
    const auto &context = *static_cast<CellGatherContext *>(raw_context);
    const uint32_t first = uint32_t(task) * context.NodesPerTask;
    const uint32_t last = std::min<uint32_t>(first + context.NodesPerTask, context.Operation.Nodes.size());
    const uint32_t dofs = context.Operation.Dofs();
    for (uint32_t node = first; node < last; ++node)
        for (uint32_t block = 0; block < context.Width; ++block) {
            std::array<double, 3> shifted{}, mass{};
            for (uint32_t entry = context.Operation.NodeOccurrenceOffsets[node];
                 entry < context.Operation.NodeOccurrenceOffsets[node + 1]; ++entry) {
                const uint32_t occurrence = context.Operation.NodeOccurrences[entry];
                const size_t offset = (size_t(occurrence >> 5) * context.Width + block) * LocalDofs + 3 * (occurrence & 31);
                for (uint32_t component = 0; component < 3; ++component) {
                    if (context.CellOutput) shifted[component] += context.CellOutput[offset + component];
                    if (context.CellMassOutput) mass[component] += context.CellMassOutput[offset + component];
                }
            }
            const size_t output = size_t(block) * dofs + 3 * node;
            for (uint32_t component = 0; component < 3; ++component) {
                if (context.Output) context.Output[output + component] = shifted[component];
                if (context.MassOutput) context.MassOutput[output + component] = mass[component];
            }
        }
}

struct ParallelTensorContext {
    const modal::FiniteCellOperator &Operation;
    const double *Input;
    double *Output, *IndependentMassOutput;
    double StiffnessScale, MassScale;
};

void ApplyTensorColumn(void *raw_context, size_t column) {
    const auto &context = *static_cast<ParallelTensorContext *>(raw_context);
    const size_t offset = column * context.Operation.Dofs();
    ApplyTensorSerial(
        context.Operation, context.Input + offset, context.Output + offset, 1,
        context.StiffnessScale, context.MassScale,
        context.IndependentMassOutput ? context.IndependentMassOutput + offset : nullptr
    );
}

// Assigns each independent right-hand side to one task.
void ApplyTensorParallel(
    const modal::FiniteCellOperator &operation, const double *input, double *output,
    uint32_t width, double stiffness_scale, double mass_scale, double *independent_mass_output = nullptr
) {
    if (!width) return;
    if (width == 1) {
        ApplyTensorSerial(operation, input, output, 1, stiffness_scale, mass_scale, independent_mass_output);
        return;
    }
    ParallelTensorContext context{operation, input, output, independent_mass_output, stiffness_scale, mass_scale};
    dispatch_apply_f(width, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &context, ApplyTensorColumn);
}

// Writes local mass and shifted `K + alpha*M` matrices for one cell.
// `mass` receives the lower triangle of the column-major nodal mass matrix.
// `shifted` receives a full column-major matrix or a row-packed lower triangle when `Packed` is true.
// A null output omits that matrix.
template<uint32_t Order, bool Packed>
void CellOperators(
    const modal::FiniteCellOperator &operation, uint32_t cell_index, double alpha,
    double *mass, double *shifted
) {
    constexpr uint32_t NodeCount{(Order + 1) * (Order + 1) * (Order + 1)}, LocalDofs{3 * NodeCount};
    constexpr size_t ShiftedValues{Packed ? size_t(LocalDofs) * (LocalDofs + 1) / 2 : size_t(LocalDofs) * LocalDofs};
    const auto &cell = operation.Cells[cell_index];
    if (mass) std::fill_n(mass, size_t(NodeCount) * NodeCount, 0.0);
    if (shifted) std::fill_n(shifted, ShiftedValues, 0.0);
    for (uint32_t quadrature = cell.QuadratureOffset; quadrature < cell.QuadratureOffset + cell.QuadratureCount; ++quadrature) {
        const auto &point = operation.Quadrature[quadrature];
        std::array<double, NodeCount> shape;
        std::array<dvec3, NodeCount> gradient;
        EvaluateBasis<Order>(point.Reference, cell.InverseHalf, shape, gradient);
        if (mass)
            for (uint32_t a = 0; a < NodeCount; ++a)
                for (uint32_t c = 0; c <= a; ++c)
                    mass[size_t(c) * NodeCount + a] += point.Weight * operation.Density * shape[a] * shape[c];
        if (!shifted) continue;
        for (uint32_t a = 0; a < NodeCount; ++a) {
            for (uint32_t c = 0; c < (Packed ? a + 1 : NodeCount); ++c) {
                const double dot = numeric::Dot(gradient[a], gradient[c]);
                for (uint32_t p = 0; p < 3; ++p) {
                    for (uint32_t q = 0; q < (Packed && c == a ? p + 1 : 3u); ++q) {
                        const uint32_t row = 3 * a + p, column = 3 * c + q;
                        shifted[Packed ? size_t(row) * (row + 1) / 2 + column : size_t(column) * LocalDofs + row] +=
                            point.Weight *
                            (operation.Lambda * gradient[a][p] * gradient[c][q] +
                             operation.Mu * gradient[a][q] * gradient[c][p] +
                             (p == q ? operation.Mu * dot + alpha * operation.Density * shape[a] * shape[c] : 0));
                    }
                }
            }
        }
    }
}

// Returns assembled lower-triangle mass and stiffness matrices over the cell basis or nested P1 corners.
// Fixed per-cell triplet slots make parallel and serial emission orders identical.
template<bool P1>
modal::FiniteCellOperator::AssembledLower Assemble(const modal::FiniteCellOperator &operation) {
    constexpr uint32_t Basis{P1 ? 1 : BasisOrder}, Nodes{(Basis + 1) * (Basis + 1) * (Basis + 1)}, Dofs{3 * Nodes};
    constexpr uint32_t MassEntries{3 * Nodes * (Nodes + 1) / 2}, StiffnessEntries{Dofs * (Dofs + 1) / 2};
    std::vector<Eigen::Triplet<double>> mass_triplets(operation.Cells.size() * MassEntries);
    std::vector<Eigen::Triplet<double>> stiffness_triplets(operation.Cells.size() * StiffnessEntries);
    struct Context {
        const modal::FiniteCellOperator &Operation;
        Eigen::Triplet<double> *Mass, *Stiffness;
    } context{operation, mass_triplets.data(), stiffness_triplets.data()};
    dispatch_apply_f(
        operation.Cells.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &context,
        [](void *raw, size_t cell_index) {
            const auto &context = *static_cast<Context *>(raw);
            const auto &cell = context.Operation.Cells[cell_index];
            std::array<uint32_t, Nodes> nodes;
            if constexpr (P1) {
                for (uint32_t z = 0; z < 2; ++z)
                    for (uint32_t y = 0; y < 2; ++y)
                        for (uint32_t x = 0; x < 2; ++x) {
                            const uint32_t local = BasisOrder * x + BasisWidth * (BasisOrder * y + BasisWidth * BasisOrder * z);
                            nodes[x + 2 * (y + 2 * z)] = context.Operation.P1Stencils[cell.Nodes[local]].Nodes[0];
                        }
            } else {
                std::copy_n(cell.Nodes.begin(), Nodes, nodes.begin());
            }
            Eigen::Matrix<double, Nodes, Nodes> mass;
            Eigen::Matrix<double, Dofs, Dofs> stiffness;
            CellOperators<Basis, false>(context.Operation, uint32_t(cell_index), 0, mass.data(), stiffness.data());
            auto *mass_triplet = context.Mass + cell_index * MassEntries;
            auto *stiffness_triplet = context.Stiffness + cell_index * StiffnessEntries;
            for (uint32_t a = 0; a < Nodes; ++a)
                for (uint32_t c = 0; c <= a; ++c)
                    for (uint32_t component = 0; component < 3; ++component) {
                        const uint32_t row = 3 * nodes[a] + component, column = 3 * nodes[c] + component;
                        *mass_triplet++ = {int(std::max(row, column)), int(std::min(row, column)), mass(a, c)};
                    }
            for (uint32_t a = 0; a < Nodes; ++a)
                for (uint32_t p = 0; p < 3; ++p)
                    for (uint32_t c = 0; c < Nodes; ++c)
                        for (uint32_t q = 0; q < 3; ++q) {
                            const uint32_t row = 3 * nodes[a] + p, column = 3 * nodes[c] + q;
                            if (row >= column) *stiffness_triplet++ = {int(row), int(column), stiffness(3 * a + p, 3 * c + q)};
                        }
        }
    );
    const uint32_t dofs = P1 ? 3 * operation.NumP1Nodes : operation.Dofs();
    modal::FiniteCellOperator::AssembledLower result{
        Eigen::SparseMatrix<double>{dofs, dofs}, Eigen::SparseMatrix<double>{dofs, dofs}
    };
    result.Mass.setFromTriplets(mass_triplets.begin(), mass_triplets.end());
    result.Stiffness.setFromTriplets(stiffness_triplets.begin(), stiffness_triplets.end());
    return result;
}

Eigen::VectorXd ShiftedDiagonal(const modal::FiniteCellOperator &operation, double alpha) {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(operation.Dofs());
    for (const auto &cell : operation.Cells) {
        for (uint32_t quadrature = cell.QuadratureOffset; quadrature < cell.QuadratureOffset + cell.QuadratureCount; ++quadrature) {
            const auto &point = operation.Quadrature[quadrature];
            std::array<double, NodeCount> shape;
            std::array<dvec3, NodeCount> gradient;
            EvaluateBasis<BasisOrder>(point.Reference, cell.InverseHalf, shape, gradient);
            for (uint32_t a = 0; a < NodeCount; ++a) {
                const double dot = numeric::Dot(gradient[a], gradient[a]);
                for (uint32_t component = 0; component < 3; ++component)
                    result[3 * cell.Nodes[a] + component] += point.Weight *
                        (alpha * operation.Density * shape[a] * shape[a] +
                         (operation.Lambda + operation.Mu) * gradient[a][component] * gradient[a][component] +
                         operation.Mu * dot);
            }
        }
    }
    return result;
}

modal::FiniteCellOperator::PackedCutOperators BuildPackedCutOperators(
    const modal::FiniteCellOperator &operation, double alpha,
    std::span<const double> packed_shifted_elements = {}
) {
    constexpr size_t PackedMass{size_t(NodeCount) * (NodeCount + 1) / 2};
    constexpr size_t PackedShifted{size_t(LocalDofs) * (LocalDofs + 1) / 2};
    const auto start = Clock::now();
    modal::FiniteCellOperator::PackedCutOperators result;
    result.Alpha = alpha;
    for (uint32_t cell = 0; cell < operation.Cells.size(); ++cell)
        (operation.Cells[cell].Cut ? result.CutCells : result.InteriorCells).push_back(cell);
    result.Mass.resize(result.CutCells.size() * PackedMass);
    result.Shifted.resize(result.CutCells.size() * PackedShifted);
    struct Context {
        const modal::FiniteCellOperator &Operation;
        modal::FiniteCellOperator::PackedCutOperators &Result;
        std::span<const double> PackedShiftedElements;
        double Alpha;
    } context{operation, result, packed_shifted_elements, alpha};
    dispatch_apply_f(
        result.CutCells.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &context,
        [](void *raw, size_t cut) {
            auto &context = *static_cast<Context *>(raw);
            const uint32_t cell = context.Result.CutCells[cut];
            const bool supplied = !context.PackedShiftedElements.empty();
            double *packed_mass = context.Result.Mass.data() + cut * PackedMass;
            double *packed_shifted = context.Result.Shifted.data() + cut * PackedShifted;
            Eigen::Matrix<double, NodeCount, NodeCount> mass;
            CellOperators<BasisOrder, true>(
                context.Operation, cell, context.Alpha, mass.data(), supplied ? nullptr : packed_shifted
            );
            for (uint32_t row = 0; row < NodeCount; ++row)
                for (uint32_t column = 0; column <= row; ++column)
                    packed_mass[size_t(row) * (row + 1) / 2 + column] = mass(row, column);
            if (supplied)
                std::copy_n(context.PackedShiftedElements.data() + size_t(cell) * PackedShifted, PackedShifted, packed_shifted);
        }
    );
    result.BuildSeconds = SecondsSince(start);
    return result;
}

constexpr uint32_t InteriorCellsPerTask{4}, GatherNodesPerTask{64};

void ApplyMassShiftedCut(
    const modal::FiniteCellOperator &operation,
    modal::FiniteCellOperator::PackedCutOperators &operators,
    const double *input, double *mass_output, double *shifted_output, uint32_t width
) {
    if (!width) return;
    constexpr size_t PackedMass{size_t(NodeCount) * (NodeCount + 1) / 2};
    constexpr size_t PackedShifted{size_t(LocalDofs) * (LocalDofs + 1) / 2};
    const size_t cell_values = size_t(operation.Cells.size()) * width * LocalDofs;
    const auto grown = [cell_values](std::vector<double> &scratch, bool wanted) {
        if (!wanted) return static_cast<double *>(nullptr);
        if (scratch.size() < cell_values) scratch.resize(cell_values);
        return scratch.data();
    };
    struct Context {
        const modal::FiniteCellOperator &Operation;
        const modal::FiniteCellOperator::PackedCutOperators &Operators;
        const double *Input;
        double *CellOutput, *CellMassOutput;
        uint32_t Width;
    } context{
        operation, operators, input, grown(operators.ShiftedCells, shifted_output),
        grown(operators.MassCells, mass_output), width
    };
    if (!operators.InteriorCells.empty()) {
        const uint32_t tasks =
            (uint32_t(operators.InteriorCells.size()) + InteriorCellsPerTask - 1) /
            InteriorCellsPerTask;
        dispatch_apply_f(
            tasks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &context,
            [](void *raw, size_t task) {
                const auto &context = *static_cast<Context *>(raw);
                const uint32_t first = uint32_t(task) * InteriorCellsPerTask;
                ApplyTensorSerial(
                    context.Operation, context.Input, nullptr, context.Width, 1,
                    context.Operators.Alpha, nullptr, first,
                    std::min(InteriorCellsPerTask, uint32_t(context.Operators.InteriorCells.size()) - first),
                    context.CellOutput, context.CellMassOutput,
                    context.Operators.InteriorCells.data()
                );
            }
        );
    }
    if (!operators.CutCells.empty()) {
        dispatch_apply_f(
            operators.CutCells.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
            &context, [](void *raw, size_t cut) {
                const auto &context = *static_cast<Context *>(raw);
                const uint32_t cell_index = context.Operators.CutCells[cut];
                const auto &cell = context.Operation.Cells[cell_index];
                const size_t local_offset = size_t(cell_index) * context.Width * LocalDofs;
                thread_local std::vector<double> expanded, local_input, mass_input, mass_result;
                expanded.resize(size_t(LocalDofs) * LocalDofs);
                local_input.resize(size_t(LocalDofs) * context.Width);
                for (uint32_t block = 0; block < context.Width; ++block)
                    for (uint32_t node = 0; node < NodeCount; ++node)
                        for (uint32_t component = 0; component < 3; ++component)
                            local_input[size_t(block) * LocalDofs + 3 * node + component] =
                                context.Input[size_t(block) * context.Operation.Dofs() + 3 * cell.Nodes[node] + component];
                // Expand the lower triangle because Accelerate matrix-matrix multiplication requires a dense column-major operand.
                const auto expand = [&](const double *packed, uint32_t size) {
                    for (uint32_t column = 0; column < size; ++column)
                        for (uint32_t row = 0; row < size; ++row) {
                            const uint32_t lower = std::max(row, column), upper = std::min(row, column);
                            expanded[size_t(column) * size + row] = packed[size_t(lower) * (lower + 1) / 2 + upper];
                        }
                };
                if (context.CellOutput) {
                    expand(context.Operators.Shifted.data() + cut * PackedShifted, LocalDofs);
                    cblas_dgemm(
                        CblasColMajor, CblasNoTrans, CblasNoTrans, LocalDofs, context.Width,
                        LocalDofs, 1, expanded.data(), LocalDofs, local_input.data(), LocalDofs,
                        0, context.CellOutput + local_offset, LocalDofs
                    );
                }
                if (!context.CellMassOutput) return;
                mass_input.resize(size_t(NodeCount) * 3 * context.Width);
                mass_result.resize(mass_input.size());
                for (uint32_t block = 0; block < context.Width; ++block)
                    for (uint32_t node = 0; node < NodeCount; ++node)
                        for (uint32_t component = 0; component < 3; ++component)
                            mass_input[size_t(3 * block + component) * NodeCount + node] =
                                local_input[size_t(block) * LocalDofs + 3 * node + component];
                expand(context.Operators.Mass.data() + cut * PackedMass, NodeCount);
                cblas_dgemm(
                    CblasColMajor, CblasNoTrans, CblasNoTrans, NodeCount, 3 * context.Width,
                    NodeCount, 1, expanded.data(), NodeCount, mass_input.data(), NodeCount, 0,
                    mass_result.data(), NodeCount
                );
                double *local_mass = context.CellMassOutput + local_offset;
                for (uint32_t block = 0; block < context.Width; ++block)
                    for (uint32_t node = 0; node < NodeCount; ++node)
                        for (uint32_t component = 0; component < 3; ++component)
                            local_mass[size_t(block) * LocalDofs + 3 * node + component] =
                                mass_result[size_t(3 * block + component) * NodeCount + node];
            }
        );
    }
    CellGatherContext gather_context{
        operation, context.CellOutput, context.CellMassOutput, shifted_output, mass_output, width, GatherNodesPerTask
    };
    const uint32_t tasks = (uint32_t(operation.Nodes.size()) + GatherNodesPerTask - 1) / GatherNodesPerTask;
    dispatch_apply_f(
        tasks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &gather_context,
        ApplyTensorCellGatherTile
    );
}

} // namespace

void modal::FiniteCellOperator::ApplyMass(const double *input, double *output, uint32_t width) const {
    ApplyTensorParallel(*this, input, output, width, 0, 1);
}

void modal::FiniteCellOperator::ApplyMassShifted(
    const double *input, double *mass_output, double *shifted_output, uint32_t width, double alpha
) const {
    ApplyTensorParallel(*this, input, shifted_output, width, 1, alpha, mass_output);
}

void modal::FiniteCellOperator::ApplyMassShiftedExpandedPackedCut(
    PackedCutOperators &operators, const double *input, double *mass_output, double *shifted_output,
    uint32_t width
) const {
    if (operators.InteriorCells.size() + operators.CutCells.size() != Cells.size())
        throw std::invalid_argument("Finite-cell packed cut operators do not match the operation.");
    ::ApplyMassShiftedCut(*this, operators, input, mass_output, shifted_output, width);
}

void modal::FiniteCellOperator::ApplyStiffness(const double *input, double *output, uint32_t width) const {
    ApplyTensorParallel(*this, input, output, width, 1, 0);
}

void modal::FiniteCellOperator::ApplyShifted(const double *input, double *output, uint32_t width, double alpha) const {
    ApplyTensorParallel(*this, input, output, width, 1, alpha);
}

void modal::FiniteCellOperator::RestrictP1(const double *input, double *output, uint32_t width) const {
    const uint32_t coarse_dofs = 3 * NumP1Nodes, fine_dofs = Dofs();
    std::fill_n(output, size_t(coarse_dofs) * width, 0.0);
    for (uint32_t block = 0; block < width; ++block)
        for (uint32_t fine = 0; fine < P1Stencils.size(); ++fine)
            for (uint32_t component = 0; component < 3; ++component)
                for (uint32_t entry = 0; entry < P1Stencils[fine].Count; ++entry)
                    output[size_t(block) * coarse_dofs + 3 * P1Stencils[fine].Nodes[entry] + component] +=
                        P1Stencils[fine].Weights[entry] * input[size_t(block) * fine_dofs + 3 * fine + component];
}

void modal::FiniteCellOperator::ProlongP1(const double *input, double *output, uint32_t width) const {
    const uint32_t coarse_dofs = 3 * NumP1Nodes, fine_dofs = Dofs();
    for (uint32_t block = 0; block < width; ++block)
        for (uint32_t fine = 0; fine < P1Stencils.size(); ++fine)
            for (uint32_t component = 0; component < 3; ++component) {
                double value{};
                for (uint32_t entry = 0; entry < P1Stencils[fine].Count; ++entry)
                    value += P1Stencils[fine].Weights[entry] *
                        input[size_t(block) * coarse_dofs + 3 * P1Stencils[fine].Nodes[entry] + component];
                output[size_t(block) * fine_dofs + 3 * fine + component] = value;
            }
}

Eigen::VectorXd modal::FiniteCellOperator::ShiftedDiagonal(double alpha) const {
    return ::ShiftedDiagonal(*this, alpha);
}

void modal::FiniteCellOperator::PackCellShiftedLower(uint32_t cell, double alpha, std::span<double> packed) const {
    const size_t local_dofs = 3 * NodesPerCell;
    if (cell >= Cells.size()) throw std::out_of_range("Finite-cell local matrix index is out of range.");
    if (packed.size() != local_dofs * (local_dofs + 1) / 2)
        throw std::invalid_argument("Finite-cell packed local matrix does not match the operation.");
    CellOperators<2, true>(*this, cell, alpha, nullptr, packed.data());
}

modal::FiniteCellOperator::PackedCutOperators modal::FiniteCellOperator::BuildPackedCutOperators(double alpha) const {
    return ::BuildPackedCutOperators(*this, alpha);
}

modal::FiniteCellOperator::PackedCutOperators modal::FiniteCellOperator::BuildPackedCutOperators(
    double alpha, std::span<const double> packed_shifted_elements
) const {
    const size_t local_dofs = 3 * NodesPerCell;
    if (packed_shifted_elements.size() != Cells.size() * local_dofs * (local_dofs + 1) / 2)
        throw std::invalid_argument("Finite-cell packed shifted elements do not match the operation.");
    return ::BuildPackedCutOperators(*this, alpha, packed_shifted_elements);
}

modal::FiniteCellOperator modal::FiniteCellOperator::WithFictitiousScale(double scale) const {
    if (!(scale > 0 && scale <= 1)) throw std::invalid_argument("Finite-cell fictitious scale must be in (0, 1].");
    FiniteCellOperator result = *this;
    for (auto &point : result.Quadrature)
        if (point.Fictitious) point.Weight *= scale / FictitiousScale;
    result.FictitiousScale = scale;
    return result;
}

modal::FiniteCellOperator::AssembledLower modal::FiniteCellOperator::AssembleLower() const {
    return Assemble<false>(*this);
}

modal::FiniteCellOperator::AssembledLower modal::FiniteCellOperator::AssembleP1Lower() const {
    return Assemble<true>(*this);
}

Eigen::SparseMatrix<double> modal::FiniteCellOperator::AssembleP1ShiftedLower(double alpha) const {
    auto assembled = AssembleP1Lower();
    return assembled.Stiffness + alpha * assembled.Mass;
}

modal::ImplicitDomain modal::MakeBoxDomain(dvec3 min, dvec3 max) {
    if (max.x <= min.x || max.y <= min.y || max.z <= min.z)
        throw std::invalid_argument("Finite-cell box bounds must have positive extent.");
    return {min, max, [center = 0.5 * (min + max), half = 0.5 * (max - min)](const dvec3 &point) {
                const dvec3 q = numeric::Abs(point - center) - half;
                const dvec3 outside = numeric::Max(q, dvec3{0});
                return numeric::Length(outside) + std::min(std::max({q.x, q.y, q.z}), 0.0); }, [min, max](const dvec3 &center, const dvec3 &half) {
                const dvec3 cell_min = center - half, cell_max = center + half;
                if (cell_min.x >= min.x && cell_min.y >= min.y && cell_min.z >= min.z && cell_max.x <= max.x && cell_max.y <= max.y && cell_max.z <= max.z)
                    return modal::DomainRegion::Inside;
                if (cell_max.x < min.x || cell_max.y < min.y || cell_max.z < min.z || cell_min.x > max.x || cell_min.y > max.y || cell_min.z > max.z)
                    return modal::DomainRegion::Outside;
                return modal::DomainRegion::Cut; }};
}

modal::ImplicitDomain modal::MakeSphereDomain(dvec3 center, double radius) {
    if (!(radius > 0)) throw std::invalid_argument("Finite-cell sphere radius must be positive.");
    return {
        center - radius,
        center + radius,
        [=](const dvec3 &point) { return numeric::Length(point - center) - radius; },
        [=](const dvec3 &cell_center, const dvec3 &half) {
            const auto [nearest, farthest] = BoxDistanceBounds(cell_center, half, center);
            if (farthest <= radius) return modal::DomainRegion::Inside;
            if (nearest >= radius) return modal::DomainRegion::Outside;
            return modal::DomainRegion::Cut;
        },
    };
}

modal::ImplicitDomain modal::MakeSphericalShellDomain(dvec3 center, double inner_radius, double outer_radius) {
    if (!(inner_radius > 0 && outer_radius > inner_radius))
        throw std::invalid_argument("Finite-cell spherical-shell radii must be positive and ordered.");
    return {
        center - outer_radius,
        center + outer_radius,
        [=](const dvec3 &point) {
            const double radius = numeric::Length(point - center);
            return std::max(inner_radius - radius, radius - outer_radius);
        },
        [=](const dvec3 &cell_center, const dvec3 &half) {
            const auto [nearest, farthest] = BoxDistanceBounds(cell_center, half, center);
            if (nearest >= inner_radius && farthest <= outer_radius) return modal::DomainRegion::Inside;
            if (farthest <= inner_radius || nearest >= outer_radius) return modal::DomainRegion::Outside;
            return modal::DomainRegion::Cut;
        },
    };
}

modal::ImplicitDomain modal::MakeCylinderDomain(dvec3 center, double radius, double length) {
    if (!(radius > 0 && length > 0))
        throw std::invalid_argument("Finite-cell cylinder radius and length must be positive.");
    const dvec3 half{radius, radius, 0.5 * length};
    return {
        center - half,
        center + half,
        [=](const dvec3 &point) {
            const dvec3 relative = point - center;
            const double radial = std::hypot(relative.x, relative.y) - radius;
            const double axial = std::abs(relative.z) - 0.5 * length;
            return std::hypot(std::max(radial, 0.0), std::max(axial, 0.0)) +
                std::min(std::max(radial, axial), 0.0);
        },
        [=](const dvec3 &cell_center, const dvec3 &cell_half) {
            const auto [nearest_radial, farthest_radial] = BoxRadialDistanceBounds(
                cell_center, cell_half, center
            );
            const double axial_distance = std::abs(cell_center.z - center.z);
            const double nearest_axial = std::max(axial_distance - cell_half.z, 0.0);
            const double farthest_axial = axial_distance + cell_half.z;
            if (farthest_radial <= radius && farthest_axial <= 0.5 * length)
                return modal::DomainRegion::Inside;
            if (nearest_radial >= radius || nearest_axial >= 0.5 * length)
                return modal::DomainRegion::Outside;
            return modal::DomainRegion::Cut;
        },
    };
}

modal::ImplicitDomain modal::MakeTriangleSurfaceDomain(std::span<const dvec3> points, std::span<const uint32_t> triangle_indices) {
    if (points.empty() || triangle_indices.empty() || triangle_indices.size() % 3)
        throw std::invalid_argument("Finite-cell surface must contain indexed triangles.");
    auto surface = std::make_shared<TriangleSurface>();
    surface->Triangles.reserve(triangle_indices.size() / 3);
    dvec3 min{std::numeric_limits<double>::infinity()}, max{-std::numeric_limits<double>::infinity()};
    for (const auto &point : points) {
        min = numeric::Min(min, point);
        max = numeric::Max(max, point);
    }
    for (size_t triangle = 0; triangle < triangle_indices.size(); triangle += 3) {
        std::array<dvec3, 3> vertices;
        for (uint32_t corner = 0; corner < 3; ++corner) {
            const uint32_t index = triangle_indices[triangle + corner];
            if (index >= points.size()) throw std::invalid_argument("Finite-cell surface index is out of range.");
            vertices[corner] = points[index];
        }
        if (numeric::Length2(numeric::Cross(vertices[1] - vertices[0], vertices[2] - vertices[0])) > 0)
            surface->Triangles.push_back(vertices);
    }
    if (surface->Triangles.empty()) throw std::invalid_argument("Finite-cell surface has no nondegenerate triangles.");
    surface->Build();
    return {
        min,
        max,
        [surface](const dvec3 &point) { return surface->SignedDistance(point); },
        [surface = std::move(surface)](const dvec3 &center, const dvec3 &half) { return surface->ClassifyBox(center, half); },
    };
}

modal::FiniteCellOperator BuildOperator(
    const modal::ImplicitDomain &domain, const AcousticMaterialProperties &material, modal::FiniteCellConfig config,
    bool moment_fitting
) {
    if (!domain.SignedDistance || config.Cells.x == 0 || config.Cells.y == 0 || config.Cells.z == 0)
        throw std::invalid_argument("Finite-cell domain and grid must be nonempty.");
    if (!(config.FictitiousScale > 0 && config.FictitiousScale <= 1) || config.PaddingCells < 0 ||
        std::abs(config.GridOffsetCells.x) > config.PaddingCells ||
        std::abs(config.GridOffsetCells.y) > config.PaddingCells || std::abs(config.GridOffsetCells.z) > config.PaddingCells)
        throw std::invalid_argument("Finite-cell configuration parameters are out of range.");

    const auto start = Clock::now();
    return Build(domain, material, config, start, moment_fitting);
}

modal::FiniteCellOperator modal::BuildFiniteCellOperator(const ImplicitDomain &domain, const AcousticMaterialProperties &material, FiniteCellConfig config) {
    return BuildOperator(domain, material, config, true);
}

std::optional<modal::FiniteCellOperator::InterpolationStencil> modal::FiniteCellOperator::InterpolationAt(dvec3 point) const {
    if (Cells.empty() || GridCells.x == 0 || GridCells.y == 0 || GridCells.z == 0) return std::nullopt;
    const dvec3 grid_coordinate = (point - GridMin) / CellStep;
    const auto coordinate = [&](uint32_t axis) {
        return std::clamp(int64_t(std::floor(grid_coordinate[axis])), int64_t{0}, int64_t(GridCells[axis]) - 1);
    };
    const std::array<int64_t, 3> base{coordinate(0), coordinate(1), coordinate(2)};
    for (int64_t dx = -1; dx <= 1; ++dx) {
        for (int64_t dy = -1; dy <= 1; ++dy) {
            for (int64_t dz = -1; dz <= 1; ++dz) {
                const std::array<int64_t, 3> candidate{base[0] + dx, base[1] + dy, base[2] + dz};
                if (candidate[0] < 0 || candidate[1] < 0 || candidate[2] < 0 || candidate[0] >= GridCells.x || candidate[1] >= GridCells.y || candidate[2] >= GridCells.z)
                    continue;
                const uint32_t background = (uint32_t(candidate[0]) * GridCells.y + uint32_t(candidate[1])) * GridCells.z + uint32_t(candidate[2]);
                const int32_t cell_index = CellAtBackgroundIndex[background];
                if (cell_index < 0) continue;
                const auto &cell = Cells[cell_index];
                const dvec3 half = 1.0 / cell.InverseHalf;
                const dvec3 center = Nodes[cell.Nodes[0]] + half;
                const dvec3 reference = (point - center) / half;
                const dvec3 absolute = numeric::Abs(reference);
                if (absolute.x > 1 + 1e-10 || absolute.y > 1 + 1e-10 || absolute.z > 1 + 1e-10) continue;
                double basis[3][3];
                for (uint32_t axis = 0; axis < 3; ++axis) {
                    const double x = reference[axis];
                    basis[axis][0] = 0.5 * x * (x - 1);
                    basis[axis][1] = 1 - x * x;
                    basis[axis][2] = 0.5 * x * (x + 1);
                }
                InterpolationStencil result{.Nodes = cell.Nodes};
                for (uint32_t z = 0; z < 3; ++z)
                    for (uint32_t y = 0; y < 3; ++y)
                        for (uint32_t x = 0; x < 3; ++x) {
                            const uint32_t node = x + 3 * (y + 3 * z);
                            result.Weights[node] = basis[0][x] * basis[1][y] * basis[2][z];
                        }
                return result;
            }
        }
    }
    return std::nullopt;
}

modal::FiniteCellOperator modal::oracle::BuildOctree(const ImplicitDomain &domain, const AcousticMaterialProperties &material, FiniteCellConfig config) {
    return BuildOperator(domain, material, config, false);
}

modal::FiniteCellCertification modal::CertifyFiniteCellEigenpairs(
    const FiniteCellOperator &operation, const Eigen::VectorXd &eigenvalues, const Eigen::MatrixXd &eigenvectors
) {
    FiniteCellCertification result;
    if (eigenvectors.rows() != operation.Dofs() || eigenvectors.cols() != eigenvalues.size()) return result;
    Eigen::MatrixXd mass(operation.Dofs(), eigenvectors.cols()), stiffness(operation.Dofs(), eigenvectors.cols());
    // Alpha zero computes independent mass and stiffness actions in one traversal.
    operation.ApplyMassShifted(eigenvectors.data(), mass.data(), stiffness.data(), uint32_t(eigenvectors.cols()), 0);
    const Eigen::MatrixXd residual = stiffness - mass * eigenvalues.asDiagonal();
    result.RelativeResiduals.resize(eigenvalues.size());
    for (Eigen::Index mode = 0; mode < eigenvalues.size(); ++mode) {
        const double scale = stiffness.col(mode).norm() + std::abs(eigenvalues[mode]) * mass.col(mode).norm();
        result.RelativeResiduals[mode] = scale == 0 ? residual.col(mode).norm() : residual.col(mode).norm() / scale;
    }
    result.MassOrthogonalityError =
        (eigenvectors.transpose() * mass - Eigen::MatrixXd::Identity(eigenvalues.size(), eigenvalues.size())).norm();
    return result;
}
