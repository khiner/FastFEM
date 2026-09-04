#include "FiniteCellBenchmarkGeometry.h"
#include "ModeShapeComparison.h"
#include "RunSuites.h"
#include "TetReference.h"
#include "audio/FiniteCell.h"
#include "audio/FiniteCellEigensolver.h"

#include <boost/ut.hpp>

#include <Eigen/QR>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <print>
#include <vector>

using namespace boost::ut;

namespace {
constexpr AcousticMaterialProperties Material{
    .Density = 1000, .YoungModulus = 1e7, .PoissonRatio = 0.25, .Alpha = 0, .Beta = 0
};

double FirstRoot(auto function, double begin, double end, double step = 0.01) {
    double previous_x = begin, previous = function(previous_x);
    for (double x = begin + step; x <= end; x += step) {
        const double value = function(x);
        if (std::isfinite(previous) && std::isfinite(value) && std::signbit(previous) != std::signbit(value)) {
            double low = previous_x, high = x, low_value = previous;
            for (uint32_t iteration = 0; iteration < 80; ++iteration) {
                const double middle = 0.5 * (low + high), middle_value = function(middle);
                if (std::signbit(low_value) == std::signbit(middle_value)) {
                    low = middle;
                    low_value = middle_value;
                } else {
                    high = middle;
                }
            }
            return 0.5 * (low + high);
        }
        previous_x = x;
        previous = value;
    }
    throw std::runtime_error("Analytical characteristic root was not bracketed.");
}

double SphericalJ(uint32_t order, double x) {
    const double j0 = std::sin(x) / x;
    if (order == 0) return j0;
    double previous = j0, current = std::sin(x) / (x * x) - std::cos(x) / x;
    for (uint32_t n = 1; n < order; ++n) {
        const double next = (2 * n + 1) * current / x - previous;
        previous = current;
        current = next;
    }
    return current;
}

double SphericalY(uint32_t order, double x) {
    const double y0 = -std::cos(x) / x;
    if (order == 0) return y0;
    double previous = y0, current = -std::cos(x) / (x * x) - std::sin(x) / x;
    for (uint32_t n = 1; n < order; ++n) {
        const double next = (2 * n + 1) * current / x - previous;
        previous = current;
        current = next;
    }
    return current;
}
double SphericalJDerivative(uint32_t order, double x) {
    return order == 0 ? -SphericalJ(1, x) : SphericalJ(order - 1, x) - (order + 1) * SphericalJ(order, x) / x;
}
double SphericalYDerivative(uint32_t order, double x) {
    return order == 0 ? -SphericalY(1, x) : SphericalY(order - 1, x) - (order + 1) * SphericalY(order, x) / x;
}

double ModifiedBesselI(uint32_t order, double x) {
    double term = order == 0 ? 1 : 0.5 * x, sum = term;
    const double quarter_x_squared = 0.25 * x * x;
    for (uint32_t k = 1; k < 100; ++k) {
        term *= quarter_x_squared / (double(k) * (k + order));
        sum += term;
        if (std::abs(term) < 1e-16 * std::abs(sum)) break;
    }
    return sum;
}

double CylindricalJ0(double x) { return ::j0(x); }
double CylindricalJ1(double x) { return ::j1(x); }
double ModifiedI0(double x) { return ModifiedBesselI(0, x); }
double ModifiedI1(double x) { return ModifiedBesselI(1, x); }

double TorsionalTractionJ(uint32_t order, double x) {
    return x * SphericalJDerivative(order, x) - SphericalJ(order, x);
}
double TorsionalTractionY(uint32_t order, double x) {
    return x * SphericalYDerivative(order, x) - SphericalY(order, x);
}
double RadialTractionJ(double x) {
    return (Material.Lambda() + 2 * Material.Mu()) * x * SphericalJDerivative(1, x) +
        2 * Material.Lambda() * SphericalJ(1, x);
}
double RadialTractionY(double x) {
    return (Material.Lambda() + 2 * Material.Mu()) * x * SphericalYDerivative(1, x) +
        2 * Material.Lambda() * SphericalY(1, x);
}

struct SphereReference {
    double RadialWaveNumber{}, TorsionalWaveNumber{};

    double RadialFrequency(double radius) const {
        const double speed = std::sqrt((Material.Lambda() + 2 * Material.Mu()) / Material.Density);
        return speed * RadialWaveNumber / (2 * std::numbers::pi * radius);
    }

    double TorsionalFrequency(double radius) const {
        const double speed = std::sqrt(Material.Mu() / Material.Density);
        return speed * TorsionalWaveNumber / (2 * std::numbers::pi * radius);
    }
};

SphereReference SolidSphereReference() {
    return {
        .RadialWaveNumber = FirstRoot(RadialTractionJ, 0.1, 12),
        .TorsionalWaveNumber = FirstRoot([](double x) { return TorsionalTractionJ(2, x); }, 0.1, 12),
    };
}

SphereReference HollowSphereReference(double radius_ratio) {
    return {
        .RadialWaveNumber = FirstRoot([=](double x) {
            return RadialTractionJ(radius_ratio * x) * RadialTractionY(x) -
                RadialTractionY(radius_ratio * x) * RadialTractionJ(x);
        },
                                      0.1, 20),
        .TorsionalWaveNumber = FirstRoot([=](double x) {
            return TorsionalTractionJ(2, radius_ratio * x) * TorsionalTractionY(2, x) -
                TorsionalTractionY(2, radius_ratio * x) * TorsionalTractionJ(2, x);
        },
                                         0.1, 20),
    };
}

finite_cell_benchmark::Surface SphereSurface(uint32_t subdivisions, double radius) {
    auto surface = finite_cell_benchmark::Icosphere(subdivisions);
    for (auto &point : surface.Points) point *= radius;
    return surface;
}

finite_cell_benchmark::Surface HollowSphereSurface(uint32_t subdivisions, double inner_radius, double outer_radius) {
    auto surface = SphereSurface(subdivisions, outer_radius);
    auto inner = SphereSurface(subdivisions, inner_radius);
    const uint32_t offset = uint32_t(surface.Points.size());
    surface.Points.insert(surface.Points.end(), inner.Points.begin(), inner.Points.end());
    for (const uint32_t point : inner.Triangles) surface.Triangles.push_back(offset + point);
    return surface;
}

finite_cell_benchmark::Surface CylinderSurface(
    double radius, double length, uint32_t segments, uint32_t axial_cells, uint32_t radial_cells = 1
) {
    finite_cell_benchmark::Surface surface;
    const auto Side = [=](uint32_t z, uint32_t segment) { return z * segments + segment % segments; };
    for (uint32_t z = 0; z <= axial_cells; ++z)
        for (uint32_t segment = 0; segment < segments; ++segment) {
            const double angle = 2 * std::numbers::pi * segment / segments;
            surface.Points.push_back({radius * std::cos(angle), radius * std::sin(angle), length * z / axial_cells - 0.5 * length});
        }
    for (uint32_t z = 0; z < axial_cells; ++z)
        for (uint32_t segment = 0; segment < segments; ++segment) {
            const uint32_t a = Side(z, segment), b = Side(z, segment + 1);
            const uint32_t c = Side(z + 1, segment + 1), d = Side(z + 1, segment);
            surface.Triangles.insert(surface.Triangles.end(), {a, b, c, a, c, d});
        }
    const uint32_t bottom = uint32_t(surface.Points.size());
    surface.Points.push_back({0, 0, -0.5 * length});
    const uint32_t top = uint32_t(surface.Points.size());
    surface.Points.push_back({0, 0, 0.5 * length});
    const uint32_t bottom_rings = uint32_t(surface.Points.size());
    const auto AddRings = [&](double z) {
        for (uint32_t ring = 1; ring < radial_cells; ++ring)
            for (uint32_t segment = 0; segment < segments; ++segment) {
                const double angle = 2 * std::numbers::pi * segment / segments, r = radius * ring / radial_cells;
                surface.Points.push_back({r * std::cos(angle), r * std::sin(angle), z});
            }
    };
    AddRings(-0.5 * length);
    const uint32_t top_rings = uint32_t(surface.Points.size());
    AddRings(0.5 * length);
    const auto Ring = [&](bool is_top, uint32_t ring, uint32_t segment) {
        if (ring == 0) return is_top ? top : bottom;
        if (ring == radial_cells) return Side(is_top ? axial_cells : 0, segment);
        return (is_top ? top_rings : bottom_rings) + (ring - 1) * segments + segment % segments;
    };
    for (uint32_t segment = 0; segment < segments; ++segment) {
        surface.Triangles.insert(surface.Triangles.end(), {
                                                              bottom,
                                                              Ring(false, 1, segment + 1),
                                                              Ring(false, 1, segment),
                                                              top,
                                                              Ring(true, 1, segment),
                                                              Ring(true, 1, segment + 1),
                                                          });
        for (uint32_t ring = 1; ring < radial_cells; ++ring) {
            const uint32_t inner = Ring(false, ring, segment), inner_next = Ring(false, ring, segment + 1);
            const uint32_t outer = Ring(false, ring + 1, segment), outer_next = Ring(false, ring + 1, segment + 1);
            const uint32_t top_inner = Ring(true, ring, segment), top_inner_next = Ring(true, ring, segment + 1);
            const uint32_t top_outer = Ring(true, ring + 1, segment), top_outer_next = Ring(true, ring + 1, segment + 1);
            surface.Triangles.insert(surface.Triangles.end(), {
                                                                  inner,
                                                                  outer_next,
                                                                  outer,
                                                                  inner,
                                                                  inner_next,
                                                                  outer_next,
                                                                  top_inner,
                                                                  top_outer,
                                                                  top_outer_next,
                                                                  top_inner,
                                                                  top_outer_next,
                                                                  top_inner_next,
                                                              });
        }
    }
    return surface;
}

std::vector<dvec3> SphereSamples(double inner_radius, double outer_radius) {
    std::vector<dvec3> result;
    constexpr std::array directions{
        dvec3{1, 0, 0},
        dvec3{0, 1, 0},
        dvec3{0, 0, 1},
        dvec3{1, 1, 0},
        dvec3{1, 0, 1},
        dvec3{0, 1, 1},
        dvec3{1, -1, 0},
        dvec3{1, 0, -1},
        dvec3{0, 1, -1},
        dvec3{1, 1, 1},
        dvec3{1, 1, -1},
        dvec3{1, -1, 1},
        dvec3{-1, 1, 1},
    };
    for (const double radius : std::array{
             std::lerp(inner_radius, outer_radius, 0.2),
             std::lerp(inner_radius, outer_radius, 0.5),
             std::lerp(inner_radius, outer_radius, 0.8),
         })
        for (const dvec3 direction : directions) result.push_back(radius * numeric::Normalize(direction));
    return result;
}

std::vector<dvec3> CylinderSamples(double radius, double length) {
    std::vector<dvec3> result;
    for (uint32_t z = 0; z < 9; ++z)
        for (const double radial : {0.25, 0.5, 0.75})
            for (uint32_t segment = 0; segment < 12; ++segment) {
                const double angle = 2 * std::numbers::pi * segment / 12;
                result.push_back({
                    radial * radius * std::cos(angle),
                    radial * radius * std::sin(angle),
                    length * (double(z) + 0.5) / 9 - 0.5 * length,
                });
            }
    return result;
}

Eigen::MatrixXd CylinderTorsionalSubspace(
    const std::vector<dvec3> &samples, double length
) {
    Eigen::MatrixXd result(3 * samples.size(), 1);
    for (uint32_t sample = 0; sample < samples.size(); ++sample) {
        const dvec3 point = samples[sample];
        const dvec3 displacement = std::sin(std::numbers::pi * point.z / length) * dvec3{-point.y, point.x, 0};
        for (uint32_t component = 0; component < 3; ++component) result(3 * sample + component, 0) = displacement[component];
    }
    return result;
}

double DiskRadialWaveNumber() {
    return FirstRoot([](double x) {
        const double derivative = CylindricalJ0(x) - CylindricalJ1(x) / x;
        return x * derivative + Material.PoissonRatio * CylindricalJ1(x);
    },
                     0.1, 12);
}

double PlateFlexuralWaveNumber() {
    return FirstRoot([](double x) {
        const double moment_j = -CylindricalJ0(x) + (1 - Material.PoissonRatio) * CylindricalJ1(x) / x;
        const double moment_i = ModifiedI0(x) - (1 - Material.PoissonRatio) * ModifiedI1(x) / x;
        return moment_j * ModifiedI1(x) - moment_i * CylindricalJ1(x);
    },
                     0.1, 12);
}

Eigen::MatrixXd DiskRadialSubspace(const std::vector<dvec3> &samples, double radius, double wave_number) {
    Eigen::MatrixXd result(3 * samples.size(), 1);
    for (uint32_t sample = 0; sample < samples.size(); ++sample) {
        const dvec3 point = samples[sample];
        const double radial_position = std::hypot(point.x, point.y);
        const dvec3 displacement = CylindricalJ1(wave_number * radial_position / radius) *
            dvec3{point.x / radial_position, point.y / radial_position, 0};
        for (uint32_t component = 0; component < 3; ++component) result(3 * sample + component, 0) = displacement[component];
    }
    return result;
}

Eigen::MatrixXd PlateFlexuralSubspace(
    const std::vector<dvec3> &samples, double radius, double wave_number
) {
    Eigen::MatrixXd result(3 * samples.size(), 1);
    const double coefficient = -CylindricalJ1(wave_number) / ModifiedI1(wave_number);
    for (uint32_t sample = 0; sample < samples.size(); ++sample) {
        const dvec3 point = samples[sample];
        const double radial_position = std::hypot(point.x, point.y), x = wave_number * radial_position / radius;
        const double transverse = CylindricalJ0(x) + coefficient * ModifiedI0(x);
        const double radial_derivative = wave_number / radius *
            (-CylindricalJ1(x) + coefficient * ModifiedI1(x));
        const dvec3 displacement{
            -point.z * radial_derivative * point.x / radial_position,
            -point.z * radial_derivative * point.y / radial_position,
            transverse,
        };
        for (uint32_t component = 0; component < 3; ++component) result(3 * sample + component, 0) = displacement[component];
    }
    return result;
}

Eigen::MatrixXd SphereRadialSubspace(
    const std::vector<dvec3> &samples, double inner_radius, double outer_radius, double wave_number
) {
    Eigen::MatrixXd result(3 * samples.size(), 1);
    double j_coefficient{1}, y_coefficient{};
    if (inner_radius > 0) {
        j_coefficient = RadialTractionY(wave_number * inner_radius / outer_radius);
        y_coefficient = -RadialTractionJ(wave_number * inner_radius / outer_radius);
    }
    for (uint32_t sample = 0; sample < samples.size(); ++sample) {
        const dvec3 point = samples[sample];
        const double radius = numeric::Length(point), x = wave_number * radius / outer_radius;
        const double radial = j_coefficient * SphericalJ(1, x) + y_coefficient * SphericalY(1, x);
        const dvec3 displacement = radial * point / radius;
        for (uint32_t component = 0; component < 3; ++component) result(3 * sample + component, 0) = displacement[component];
    }
    return result;
}

Eigen::MatrixXd SphereTorsionalSubspace(
    const std::vector<dvec3> &samples, double inner_radius, double outer_radius, double wave_number
) {
    Eigen::MatrixXd result(3 * samples.size(), 5);
    double j_coefficient{1}, y_coefficient{};
    if (inner_radius > 0) {
        j_coefficient = TorsionalTractionY(2, wave_number * inner_radius / outer_radius);
        y_coefficient = -TorsionalTractionJ(2, wave_number * inner_radius / outer_radius);
    }
    for (uint32_t sample = 0; sample < samples.size(); ++sample) {
        const dvec3 point = samples[sample];
        const double radius = numeric::Length(point), x = wave_number * radius / outer_radius;
        const double radial = j_coefficient * SphericalJ(2, x) + y_coefficient * SphericalY(2, x);
        const std::array gradients{
            dvec3{point.y, point.x, 0},
            dvec3{point.z, 0, point.x},
            dvec3{0, point.z, point.y},
            dvec3{2 * point.x, -2 * point.y, 0},
            dvec3{-2 * point.x, -2 * point.y, 4 * point.z},
        };
        for (uint32_t basis = 0; basis < gradients.size(); ++basis) {
            const dvec3 displacement = radial * numeric::Cross(point, gradients[basis]) / (radius * radius);
            for (uint32_t component = 0; component < 3; ++component)
                result(3 * sample + component, basis) = displacement[component];
        }
    }
    return result;
}

struct SampledFinite {
    Eigen::VectorXd Values;
    Eigen::MatrixXd Modes;
    double Residual{}, Volume{};
    uint32_t Dofs{}, Iterations{};
    uint64_t QuadraturePoints{};
};

SampledFinite SolveFiniteAndSample(
    const modal::ImplicitDomain &domain, uvec3 cells, uint32_t count, const std::vector<dvec3> &samples,
    uint32_t cut_depth = 3
) {
    const auto finite = modal::BuildFiniteCellOperator(
        domain, Material,
        {.Cells = cells, .CutDepth = cut_depth, .FictitiousScale = 1e-8, .PaddingCells = 0.25}
    );
    const double shift = std::pow(2 * std::numbers::pi * 20, 2);
    const auto modes = modal::SolveFiniteCellEigenpairs(finite, count, shift, 1e-8, 300);
    expect(modes.Eigenvalues.size() == count);
    if (modes.Eigenvalues.size() != count) return {};
    const auto modes_certification = modal::CertifyFiniteCellEigenpairs(finite, modes.Eigenvalues, modes.Eigenvectors);
    std::vector<finite_cell_benchmark::InterpolationStencil> stencils;
    for (const dvec3 point : samples) {
        const auto stencil = finite_cell_benchmark::FiniteCellStencil(finite, point);
        expect(bool(stencil));
        if (stencil) stencils.push_back(*stencil);
    }
    return {
        .Values = modes.Eigenvalues,
        .Modes = finite_cell_benchmark::SampleModes(stencils, modes.Eigenvectors, 6),
        .Residual = modes_certification.RelativeResiduals.tail(count - 6).maxCoeff(),
        .Volume = finite.Profile.PhysicalVolume,
        .Dofs = finite.Dofs(),
        .Iterations = modes.Iterations,
        .QuadraturePoints = finite.Profile.QuadraturePoints,
    };
}

struct SampledPair {
    SampledFinite Finite;
    Eigen::VectorXd TetValues;
    Eigen::MatrixXd TetModes;
    double TetResidual{};
    uint32_t TetDofs{};
};

SampledPair SolveAndSample(
    const finite_cell_benchmark::Surface &surface, uvec3 finite_cells, uvec3 tet_cells, uint32_t count,
    const std::vector<dvec3> &samples, uint32_t cut_depth = 3, std::span<const dvec3> holes = {}
) {
    const auto domain = modal::MakeTriangleSurfaceDomain(surface.Points, surface.Triangles);
    SampledFinite finite = SolveFiniteAndSample(domain, finite_cells, count, samples, cut_depth);
    const auto mesh = finite_cell_benchmark::TetrahedralReference(
        surface, tet_cells.x, tet_cells.y, tet_cells.z, holes
    );
    const modal::Tet10Assembler tet{mesh, Material};
    const double shift = std::pow(2 * std::numbers::pi * 20, 2);
    const auto tet_modes = SolveTetReference(mesh, Material, count, shift, 1e-8, 200);
    expect(tet_modes.Eigenvalues.size() == count);
    if (tet_modes.Eigenvalues.size() != count) return {};
    std::vector<finite_cell_benchmark::InterpolationStencil> tet_stencils;
    for (const dvec3 point : samples) {
        const auto tet_stencil = finite_cell_benchmark::Tet10Stencil(mesh, tet, point);
        expect(bool(tet_stencil));
        if (tet_stencil) tet_stencils.push_back(*tet_stencil);
    }
    return {
        .Finite = std::move(finite),
        .TetValues = tet_modes.Eigenvalues,
        .TetModes = finite_cell_benchmark::SampleModes(tet_stencils, tet_modes.Eigenvectors, 6),
        .TetResidual = tet_modes.RelativeResidual,
        .TetDofs = tet.Dofs(),
    };
}

struct ModeMatch {
    double Frequency{}, MaximumFrequencyError{}, MinimumSubspaceMac{};
};

ModeMatch MatchModes(
    const Eigen::VectorXd &eigenvalues, const Eigen::MatrixXd &sampled, const Eigen::MatrixXd &analytical,
    double exact_frequency
) {
    Eigen::HouseholderQR<Eigen::MatrixXd> analytical_qr{analytical};
    const Eigen::MatrixXd analytical_q = analytical_qr.householderQ() *
        Eigen::MatrixXd::Identity(analytical.rows(), analytical.cols());
    std::vector<std::pair<double, Eigen::Index>> scores;
    for (Eigen::Index mode = 0; mode < sampled.cols(); ++mode) {
        const Eigen::VectorXd normalized = sampled.col(mode).normalized();
        const double frequency = std::sqrt(std::max(0.0, eigenvalues[mode + 6])) / (2 * std::numbers::pi);
        const double relative_frequency = (frequency - exact_frequency) / exact_frequency;
        const double frequency_weight = std::exp(-0.5 * std::pow(relative_frequency / 0.1, 2));
        scores.emplace_back(frequency_weight * (analytical_q.transpose() * normalized).squaredNorm(), mode);
    }
    std::ranges::sort(scores, std::greater{}, &std::pair<double, Eigen::Index>::first);
    Eigen::MatrixXd selected(sampled.rows(), analytical.cols());
    ModeMatch result;
    for (Eigen::Index column = 0; column < analytical.cols(); ++column) {
        selected.col(column) = sampled.col(scores[column].second).normalized();
        const double frequency = std::sqrt(std::max(0.0, eigenvalues[scores[column].second + 6])) /
            (2 * std::numbers::pi);
        result.Frequency += frequency / analytical.cols();
        result.MaximumFrequencyError = std::max(result.MaximumFrequencyError, std::abs(frequency / exact_frequency - 1));
    }
    Eigen::HouseholderQR<Eigen::MatrixXd> selected_qr{selected};
    const Eigen::MatrixXd selected_q = selected_qr.householderQ() *
        Eigen::MatrixXd::Identity(selected.rows(), selected.cols());
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd{analytical_q.transpose() * selected_q};
    result.MinimumSubspaceMac = std::pow(svd.singularValues().minCoeff(), 2);
    return result;
}
} // namespace

suite AnalyticalModalTests = [] {
    "solid sphere tracks exact radial and torsional Lamb eigenspaces"_test = [] {
        constexpr double radius{0.1};
        const SphereReference reference = SolidSphereReference();
        const auto samples = SphereSamples(0, 0.72 * radius);
        const Eigen::MatrixXd radial = SphereRadialSubspace(samples, 0, radius, reference.RadialWaveNumber);
        const Eigen::MatrixXd torsional = SphereTorsionalSubspace(samples, 0, radius, reference.TorsionalWaveNumber);
        std::array<double, 3> finite_errors{}, exact_errors{}, tet_errors{};
        for (uint32_t level = 0; level < 3; ++level) {
            const uint32_t subdivisions = level + 1, resolution = 6 + 2 * level;
            const auto result = SolveAndSample(
                SphereSurface(subdivisions, radius), uvec3{resolution}, uvec3{resolution}, 60, samples
            );
            const auto exact = SolveFiniteAndSample(modal::MakeSphereDomain({}, radius), uvec3{resolution}, 40, samples);
            const auto finite_radial = MatchModes(result.Finite.Values, result.Finite.Modes, radial, reference.RadialFrequency(radius));
            const auto exact_radial = MatchModes(exact.Values, exact.Modes, radial, reference.RadialFrequency(radius));
            const auto tet_radial = MatchModes(result.TetValues, result.TetModes, radial, reference.RadialFrequency(radius));
            const auto finite_torsional = MatchModes(result.Finite.Values, result.Finite.Modes, torsional, reference.TorsionalFrequency(radius));
            const auto exact_torsional = MatchModes(exact.Values, exact.Modes, torsional, reference.TorsionalFrequency(radius));
            const auto tet_torsional = MatchModes(result.TetValues, result.TetModes, torsional, reference.TorsionalFrequency(radius));
            finite_errors[level] = std::max(finite_radial.MaximumFrequencyError, finite_torsional.MaximumFrequencyError);
            exact_errors[level] = std::max(exact_radial.MaximumFrequencyError, exact_torsional.MaximumFrequencyError);
            tet_errors[level] = std::max(tet_radial.MaximumFrequencyError, tet_torsional.MaximumFrequencyError);
            std::println(
                "solid sphere s{}: tet/finite dofs={}/{}, radial exact={:.3f} Hz tet={:.3f} ({:.3f}%) finite={:.3f} ({:.3f}%), "
                "torsional exact={:.3f} Hz tet={:.3f} ({:.3f}%, MAC {:.6f}) finite={:.3f} ({:.3f}%, MAC {:.6f})",
                subdivisions, result.TetDofs, result.Finite.Dofs, reference.RadialFrequency(radius), tet_radial.Frequency,
                100 * tet_radial.MaximumFrequencyError, finite_radial.Frequency, 100 * finite_radial.MaximumFrequencyError,
                reference.TorsionalFrequency(radius), tet_torsional.Frequency, 100 * tet_torsional.MaximumFrequencyError,
                tet_torsional.MinimumSubspaceMac, finite_torsional.Frequency, 100 * finite_torsional.MaximumFrequencyError,
                finite_torsional.MinimumSubspaceMac
            );
            std::println(
                "  residuals: tet={:.3e} finite={:.3e}; exact domain: dofs={} quadrature={} iterations={} "
                "residual={:.3e} volume={:.4f}% "
                "radial={:.3f} Hz ({:.4f}%, MAC {:.6f}) "
                "torsional={:.3f} Hz ({:.4f}%, MAC {:.6f})",
                result.TetResidual, result.Finite.Residual, exact.Dofs, exact.QuadraturePoints, exact.Iterations,
                exact.Residual,
                100 * std::abs(exact.Volume / (4 * std::numbers::pi * std::pow(radius, 3) / 3) - 1),
                exact_radial.Frequency, 100 * exact_radial.MaximumFrequencyError, exact_radial.MinimumSubspaceMac,
                exact_torsional.Frequency, 100 * exact_torsional.MaximumFrequencyError,
                exact_torsional.MinimumSubspaceMac
            );
            expect(result.TetResidual < 1e-7);
            expect(result.Finite.Residual < 1e-7);
            expect(exact.Residual < 1e-7);
            expect(tet_radial.MinimumSubspaceMac > 0.98);
            expect(finite_radial.MinimumSubspaceMac > 0.98);
            expect(exact_radial.MinimumSubspaceMac > 0.98);
            expect(tet_torsional.MinimumSubspaceMac > 0.98);
            expect(finite_torsional.MinimumSubspaceMac > 0.98);
            expect(exact_torsional.MinimumSubspaceMac > 0.98);
        }
        expect(tet_errors.back() < 0.04);
        expect(finite_errors.back() < 0.06);
        expect(exact_errors.back() < 0.03);
        expect(tet_errors.back() < tet_errors.front());
        expect(finite_errors.back() < finite_errors.front());
        expect(exact_errors.back() < exact_errors.front());
    };

    "hollow sphere tracks exact inner and outer traction-free eigenspaces"_test = [] {
        constexpr double inner_radius{0.055}, outer_radius{0.1};
        const SphereReference reference = HollowSphereReference(inner_radius / outer_radius);
        const auto samples = SphereSamples(0.06, 0.075);
        const Eigen::MatrixXd radial = SphereRadialSubspace(
            samples, inner_radius, outer_radius, reference.RadialWaveNumber
        );
        const Eigen::MatrixXd torsional = SphereTorsionalSubspace(
            samples, inner_radius, outer_radius, reference.TorsionalWaveNumber
        );
        const std::array holes{dvec3{0}};
        std::array<double, 3> finite_errors{}, exact_errors{}, tet_errors{};
        for (uint32_t level = 0; level < 3; ++level) {
            const uint32_t subdivisions = level + 1, resolution = 6 + 2 * level;
            const auto result = SolveAndSample(
                HollowSphereSurface(subdivisions, inner_radius, outer_radius), uvec3{resolution}, uvec3{resolution},
                60, samples, 3, holes
            );
            const auto exact = SolveFiniteAndSample(
                modal::MakeSphericalShellDomain({}, inner_radius, outer_radius), uvec3{resolution}, 40, samples
            );
            const auto finite_radial = MatchModes(
                result.Finite.Values, result.Finite.Modes, radial, reference.RadialFrequency(outer_radius)
            );
            const auto tet_radial = MatchModes(
                result.TetValues, result.TetModes, radial, reference.RadialFrequency(outer_radius)
            );
            const auto exact_radial = MatchModes(
                exact.Values, exact.Modes, radial, reference.RadialFrequency(outer_radius)
            );
            const auto finite_torsional = MatchModes(
                result.Finite.Values, result.Finite.Modes, torsional, reference.TorsionalFrequency(outer_radius)
            );
            const auto tet_torsional = MatchModes(
                result.TetValues, result.TetModes, torsional, reference.TorsionalFrequency(outer_radius)
            );
            const auto exact_torsional = MatchModes(
                exact.Values, exact.Modes, torsional, reference.TorsionalFrequency(outer_radius)
            );
            finite_errors[level] = std::max(finite_radial.MaximumFrequencyError, finite_torsional.MaximumFrequencyError);
            exact_errors[level] = std::max(exact_radial.MaximumFrequencyError, exact_torsional.MaximumFrequencyError);
            tet_errors[level] = std::max(tet_radial.MaximumFrequencyError, tet_torsional.MaximumFrequencyError);
            std::println(
                "hollow sphere s{}: tet/finite dofs={}/{}, radial exact={:.3f} Hz tet={:.3f} ({:.3f}%, MAC {:.6f}) "
                "finite={:.3f} ({:.3f}%, MAC {:.6f}), "
                "torsional exact={:.3f} Hz tet={:.3f} ({:.3f}%, MAC {:.6f}) finite={:.3f} ({:.3f}%, MAC {:.6f})",
                subdivisions, result.TetDofs, result.Finite.Dofs, reference.RadialFrequency(outer_radius),
                tet_radial.Frequency, 100 * tet_radial.MaximumFrequencyError, tet_radial.MinimumSubspaceMac,
                finite_radial.Frequency, 100 * finite_radial.MaximumFrequencyError,
                finite_radial.MinimumSubspaceMac, reference.TorsionalFrequency(outer_radius),
                tet_torsional.Frequency, 100 * tet_torsional.MaximumFrequencyError, tet_torsional.MinimumSubspaceMac,
                finite_torsional.Frequency, 100 * finite_torsional.MaximumFrequencyError,
                finite_torsional.MinimumSubspaceMac
            );
            std::println(
                "  residuals: tet={:.3e} finite={:.3e}; exact domain: dofs={} quadrature={} iterations={} "
                "residual={:.3e} volume={:.4f}% "
                "radial={:.3f} Hz ({:.4f}%, MAC {:.6f}) "
                "torsional={:.3f} Hz ({:.4f}%, MAC {:.6f})",
                result.TetResidual, result.Finite.Residual, exact.Dofs, exact.QuadraturePoints, exact.Iterations,
                exact.Residual,
                100 * std::abs(exact.Volume / (4 * std::numbers::pi * (std::pow(outer_radius, 3) - std::pow(inner_radius, 3)) / 3) - 1),
                exact_radial.Frequency, 100 * exact_radial.MaximumFrequencyError, exact_radial.MinimumSubspaceMac,
                exact_torsional.Frequency, 100 * exact_torsional.MaximumFrequencyError,
                exact_torsional.MinimumSubspaceMac
            );
            expect(result.TetResidual < 1e-7);
            expect(result.Finite.Residual < 1e-7);
            expect(exact.Residual < 1e-7);
            expect(tet_radial.MinimumSubspaceMac > 0.98);
            expect(finite_radial.MinimumSubspaceMac > 0.98);
            expect(exact_radial.MinimumSubspaceMac > 0.98);
            expect(tet_torsional.MinimumSubspaceMac > 0.98);
            expect(finite_torsional.MinimumSubspaceMac > 0.98);
            expect(exact_torsional.MinimumSubspaceMac > 0.98);
        }
        expect(tet_errors.back() < 0.05);
        expect(finite_errors.back() < 0.07);
        expect(exact_errors.back() < 0.03);
        expect(tet_errors.back() < tet_errors.front());
        expect(finite_errors.back() < finite_errors.front());
        expect(exact_errors.back() < exact_errors.front());
    };

    "finite cylinder tracks its exact traction-free torsional mode"_test = [] {
        constexpr double radius{0.05}, length{0.3};
        constexpr std::array segments{24u, 48u, 96u};
        constexpr std::array cells{uvec3{4, 4, 12}, uvec3{5, 5, 18}, uvec3{6, 6, 24}};
        const double exact_frequency = std::sqrt(Material.Mu() / Material.Density) / (2 * length);
        const auto samples = CylinderSamples(radius, length);
        const Eigen::MatrixXd torsional = CylinderTorsionalSubspace(samples, length);
        std::array<double, 3> finite_errors{}, exact_errors{}, tet_errors{};
        for (uint32_t level = 0; level < segments.size(); ++level) {
            const auto result = SolveAndSample(
                CylinderSurface(radius, length, segments[level], cells[level].z), cells[level], cells[level], 30,
                samples
            );
            const auto exact = SolveFiniteAndSample(
                modal::MakeCylinderDomain({}, radius, length), cells[level], 30, samples
            );
            const auto finite = MatchModes(result.Finite.Values, result.Finite.Modes, torsional, exact_frequency);
            const auto exact_match = MatchModes(exact.Values, exact.Modes, torsional, exact_frequency);
            const auto tet = MatchModes(result.TetValues, result.TetModes, torsional, exact_frequency);
            finite_errors[level] = finite.MaximumFrequencyError;
            exact_errors[level] = exact_match.MaximumFrequencyError;
            tet_errors[level] = tet.MaximumFrequencyError;
            std::println(
                "cylinder {} segments: tet/finite dofs={}/{}, exact={:.3f} Hz tet={:.3f} ({:.3f}%, MAC {:.6f}) "
                "finite={:.3f} ({:.3f}%, MAC {:.6f}) cross={:.3f}%",
                segments[level], result.TetDofs, result.Finite.Dofs, exact_frequency, tet.Frequency,
                100 * tet.MaximumFrequencyError, tet.MinimumSubspaceMac, finite.Frequency,
                100 * finite.MaximumFrequencyError, finite.MinimumSubspaceMac,
                100 * std::abs(finite.Frequency / tet.Frequency - 1)
            );
            std::println(
                "  residuals: tet={:.3e} finite={:.3e}; exact domain: dofs={} quadrature={} iterations={} "
                "residual={:.3e} volume={:.4f}% "
                "frequency={:.3f} Hz ({:.4f}%, MAC {:.6f})",
                result.TetResidual, result.Finite.Residual, exact.Dofs, exact.QuadraturePoints, exact.Iterations,
                exact.Residual,
                100 * std::abs(exact.Volume / (std::numbers::pi * radius * radius * length) - 1),
                exact_match.Frequency, 100 * exact_match.MaximumFrequencyError, exact_match.MinimumSubspaceMac
            );
            expect(result.TetResidual < 1e-7);
            expect(result.Finite.Residual < 1e-7);
            expect(exact.Residual < 1e-7);
            expect(tet.MinimumSubspaceMac > 0.999);
            expect(finite.MinimumSubspaceMac > 0.999);
            expect(exact_match.MinimumSubspaceMac > 0.999);
        }
        expect(tet_errors.back() < 0.02);
        expect(finite_errors.back() < 0.03);
        expect(exact_errors.back() < 0.02);
        expect(tet_errors.back() < tet_errors.front());
        expect(finite_errors.back() < finite_errors.front());
        expect(exact_errors.back() < exact_errors.front());
    };

    "thin circular disks approach plane-stress and Kirchhoff-Love modes"_test = [] {
        constexpr double radius{0.1};
        constexpr std::array thicknesses{0.02, 0.01, 0.005};
        constexpr std::array segments{48u, 64u, 96u};
        constexpr std::array radial_cells{6u, 10u, 16u};
        constexpr std::array finite_cells{uvec3{12, 12, 2}, uvec3{16, 16, 2}, uvec3{20, 20, 2}};
        constexpr std::array tet_cells{uvec3{16, 16, 4}, uvec3{20, 20, 8}, uvec3{24, 24, 16}};
        const double radial_wave_number = DiskRadialWaveNumber();
        const double flexural_wave_number = PlateFlexuralWaveNumber();
        const double radial_frequency = radial_wave_number / (2 * std::numbers::pi * radius) *
            std::sqrt(Material.YoungModulus / (Material.Density * (1 - std::pow(Material.PoissonRatio, 2))));
        std::array<double, 3> finite_radial_errors{}, exact_radial_errors{}, tet_radial_errors{};
        std::array<double, 3> finite_flexural_errors{}, exact_flexural_errors{}, tet_flexural_errors{};
        for (uint32_t level = 0; level < thicknesses.size(); ++level) {
            const double thickness = thicknesses[level];
            const double flexural_frequency = std::pow(flexural_wave_number / radius, 2) * thickness /
                (2 * std::numbers::pi) * std::sqrt(Material.YoungModulus / (12 * Material.Density * (1 - std::pow(Material.PoissonRatio, 2))));
            const auto samples = CylinderSamples(radius, thickness);
            const Eigen::MatrixXd radial = DiskRadialSubspace(samples, radius, radial_wave_number);
            const Eigen::MatrixXd flexural = PlateFlexuralSubspace(samples, radius, flexural_wave_number);
            const auto result = SolveAndSample(
                CylinderSurface(radius, thickness, segments[level], 2, radial_cells[level]), finite_cells[level],
                tet_cells[level], 60, samples
            );
            const auto exact = SolveFiniteAndSample(
                modal::MakeCylinderDomain({}, radius, thickness), finite_cells[level], 60, samples
            );
            const auto finite_radial = MatchModes(result.Finite.Values, result.Finite.Modes, radial, radial_frequency);
            const auto exact_radial = MatchModes(exact.Values, exact.Modes, radial, radial_frequency);
            const auto tet_radial = MatchModes(result.TetValues, result.TetModes, radial, radial_frequency);
            const auto finite_flexural = MatchModes(
                result.Finite.Values, result.Finite.Modes, flexural, flexural_frequency
            );
            const auto exact_flexural = MatchModes(exact.Values, exact.Modes, flexural, flexural_frequency);
            const auto tet_flexural = MatchModes(result.TetValues, result.TetModes, flexural, flexural_frequency);
            finite_radial_errors[level] = finite_radial.MaximumFrequencyError;
            exact_radial_errors[level] = exact_radial.MaximumFrequencyError;
            tet_radial_errors[level] = tet_radial.MaximumFrequencyError;
            finite_flexural_errors[level] = finite_flexural.MaximumFrequencyError;
            exact_flexural_errors[level] = exact_flexural.MaximumFrequencyError;
            tet_flexural_errors[level] = tet_flexural.MaximumFrequencyError;
            std::println(
                "disk h/R={:.3f}: tet/finite dofs={}/{}, plane-stress exact={:.3f} Hz tet={:.3f} "
                "({:.3f}%, MAC {:.6f}) finite={:.3f} ({:.3f}%, MAC {:.6f}), Kirchhoff-Love exact={:.3f} Hz "
                "tet={:.3f} ({:.3f}%, MAC {:.6f}) finite={:.3f} ({:.3f}%, MAC {:.6f})",
                thickness / radius, result.TetDofs, result.Finite.Dofs, radial_frequency, tet_radial.Frequency,
                100 * tet_radial.MaximumFrequencyError, tet_radial.MinimumSubspaceMac, finite_radial.Frequency,
                100 * finite_radial.MaximumFrequencyError, finite_radial.MinimumSubspaceMac, flexural_frequency,
                tet_flexural.Frequency, 100 * tet_flexural.MaximumFrequencyError, tet_flexural.MinimumSubspaceMac,
                finite_flexural.Frequency, 100 * finite_flexural.MaximumFrequencyError,
                finite_flexural.MinimumSubspaceMac
            );
            std::println(
                "  residuals: tet={:.3e} finite={:.3e}; exact domain: dofs={} quadrature={} iterations={} "
                "residual={:.3e} volume={:.4f}% "
                "plane-stress={:.3f} Hz ({:.4f}%, MAC {:.6f}) "
                "Kirchhoff-Love={:.3f} Hz ({:.4f}%, MAC {:.6f})",
                result.TetResidual, result.Finite.Residual, exact.Dofs, exact.QuadraturePoints, exact.Iterations,
                exact.Residual,
                100 * std::abs(exact.Volume / (std::numbers::pi * radius * radius * thickness) - 1),
                exact_radial.Frequency, 100 * exact_radial.MaximumFrequencyError, exact_radial.MinimumSubspaceMac,
                exact_flexural.Frequency, 100 * exact_flexural.MaximumFrequencyError,
                exact_flexural.MinimumSubspaceMac
            );
            expect(result.TetResidual < 1e-7);
            expect(result.Finite.Residual < 1e-7);
            expect(exact.Residual < 1e-7);
            expect(tet_radial.MinimumSubspaceMac > 0.99);
            expect(finite_radial.MinimumSubspaceMac > 0.99);
            expect(exact_radial.MinimumSubspaceMac > 0.99);
            expect(tet_flexural.MinimumSubspaceMac > 0.99);
            expect(finite_flexural.MinimumSubspaceMac > 0.99);
            expect(exact_flexural.MinimumSubspaceMac > 0.99);
        }
        expect(tet_radial_errors.back() < 0.04);
        expect(finite_radial_errors.back() < 0.04);
        expect(exact_radial_errors.back() < 0.04);
        expect(tet_flexural_errors.back() < 0.08);
        expect(finite_flexural_errors.back() < 0.08);
        expect(exact_flexural_errors.back() < 0.08);
        expect(tet_flexural_errors.back() < tet_flexural_errors.front());
        expect(finite_flexural_errors.back() < finite_flexural_errors.front());
        expect(exact_flexural_errors.back() < exact_flexural_errors.front());
    };

    "exact implicit domains converge independently in cut depth"_test = [] {
        constexpr double sphere_radius{0.1};
        const SphereReference sphere_reference = SolidSphereReference();
        const auto sphere_samples = SphereSamples(0, 0.72 * sphere_radius);
        const Eigen::MatrixXd sphere_radial = SphereRadialSubspace(
            sphere_samples, 0, sphere_radius, sphere_reference.RadialWaveNumber
        );
        const Eigen::MatrixXd sphere_torsional = SphereTorsionalSubspace(
            sphere_samples, 0, sphere_radius, sphere_reference.TorsionalWaveNumber
        );
        std::array<double, 3> sphere_errors{}, sphere_volume_errors{};
        for (uint32_t depth = 1; depth <= 3; ++depth) {
            const auto result = SolveFiniteAndSample(
                modal::MakeSphereDomain({}, sphere_radius), uvec3{6}, 40, sphere_samples, depth
            );
            const auto radial = MatchModes(
                result.Values, result.Modes, sphere_radial, sphere_reference.RadialFrequency(sphere_radius)
            );
            const auto torsional = MatchModes(
                result.Values, result.Modes, sphere_torsional, sphere_reference.TorsionalFrequency(sphere_radius)
            );
            sphere_errors[depth - 1] = std::max(radial.MaximumFrequencyError, torsional.MaximumFrequencyError);
            sphere_volume_errors[depth - 1] = std::abs(
                result.Volume / (4 * std::numbers::pi * std::pow(sphere_radius, 3) / 3) - 1
            );
            std::println(
                "exact sphere depth {}: quadrature={} volume={:.4f}% radial={:.4f}% torsional={:.4f}% residual={:.3e}",
                depth, result.QuadraturePoints, 100 * sphere_volume_errors[depth - 1],
                100 * radial.MaximumFrequencyError, 100 * torsional.MaximumFrequencyError, result.Residual
            );
            expect(result.Residual < 1e-7);
            expect(radial.MinimumSubspaceMac > 0.98);
            expect(torsional.MinimumSubspaceMac > 0.98);
        }
        expect(sphere_errors.back() < sphere_errors.front());
        expect(sphere_volume_errors.back() < sphere_volume_errors.front());

        constexpr double disk_radius{0.1}, disk_thickness{0.02};
        const double flexural_wave_number = PlateFlexuralWaveNumber();
        const double flexural_frequency = std::pow(flexural_wave_number / disk_radius, 2) * disk_thickness /
            (2 * std::numbers::pi) * std::sqrt(Material.YoungModulus / (12 * Material.Density * (1 - std::pow(Material.PoissonRatio, 2))));
        const auto disk_samples = CylinderSamples(disk_radius, disk_thickness);
        const Eigen::MatrixXd disk_flexural = PlateFlexuralSubspace(
            disk_samples, disk_radius, flexural_wave_number
        );
        std::array<double, 4> disk_volume_errors{}, disk_flexural_frequencies{};
        for (uint32_t depth = 1; depth <= 4; ++depth) {
            const auto result = SolveFiniteAndSample(
                modal::MakeCylinderDomain({}, disk_radius, disk_thickness), uvec3{8, 8, 2}, 24, disk_samples,
                depth
            );
            const auto flexural = MatchModes(result.Values, result.Modes, disk_flexural, flexural_frequency);
            disk_volume_errors[depth - 1] = std::abs(
                result.Volume / (std::numbers::pi * disk_radius * disk_radius * disk_thickness) - 1
            );
            disk_flexural_frequencies[depth - 1] = flexural.Frequency;
            std::println(
                "exact disk depth {}: quadrature={} volume={:.4f}% flexural={:.3f} Hz ({:.4f}%) residual={:.3e}",
                depth, result.QuadraturePoints, 100 * disk_volume_errors[depth - 1], flexural.Frequency,
                100 * flexural.MaximumFrequencyError, result.Residual
            );
            expect(result.Residual < 1e-7);
            expect(flexural.MinimumSubspaceMac > 0.99);
        }
        expect(disk_volume_errors.back() < disk_volume_errors.front());
        expect(std::abs(disk_flexural_frequencies[3] / disk_flexural_frequencies[2] - 1) < std::abs(disk_flexural_frequencies[2] / disk_flexural_frequencies[1] - 1));
    };
};

int main() { return RunSuites(); }
