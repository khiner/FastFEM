#include "LoadObj.h"
#include "RunSuites.h"
#include "StructuredBar.h"
#include "ValidateTetMesh.h"
#include "audio/AcousticMaterialProperties.h"
#include "audio/BlockSparseCholesky.h"
#include "audio/CholeskyShiftInvert.h"
#include "audio/SparseCholesky.h"
#include "audio/Tet10Assembler.h"
#include "audio/mesh2modes.h"
#include "mesh/Tets.h"

#include <Eigen/Eigenvalues>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymGEigsShiftSolver.h>
#include <boost/ut.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <numbers>
#include <print>
#include <random>
#include <vector>

using namespace boost::ut;

namespace {
// Free-free rectangular prism whose mode families have closed forms (with Poisson's ratio 0):
//   Longitudinal: f_n = n*sqrt(E/rho)/(2L)
//   Torsional (square section): f_n = n*sqrt(G*J/(rho*Ip))/(2L), G = E/2, J = 0.140577*a^4, Ip = a^4/6
//   Bending (Euler-Bernoulli): f_i = (bL)_i^2/(2*pi) * sqrt(E/rho)*r_g/L^2, (bL) = {4.73004, 7.85320, 10.99561},
//     with r_g the section's radius of gyration about the bending axis (thickness/sqrt(12)).
struct Bar {
    double Length, Width, Thickness; // x, y, z extents, meters
    AcousticMaterialProperties Material;
};

constexpr double BendingBL[]{4.73004074, 7.85320462, 10.9956078};

enum class Family {
    Longitudinal,
    Torsional,
    Bending, // Lateral translation in either plane (square-section pairs are degenerate and may mix planes)
    BendingY, // Lateral translation dominantly along y
    BendingZ, // Lateral translation dominantly along z
    Other,
};

// Classify a mode by its shape's kinetic-energy fractions. Torsion is measured as the energy of the
// best-fit rigid rotation of each cross-section slice about the bar axis, so lateral translation
// (which also moves tangentially relative to the axis) does not read as torsion.
Family Classify(const ModalModes &modes, uint32_t mode, const Bar &bar, int nx) {
    double axial = 0, lateral_y = 0, lateral_z = 0, total = 0;
    std::map<int, std::pair<double, double>> slices; // x index -> (sum of r x u along the axis, sum of r^2)
    for (size_t v = 0; v < modes.Positions.size(); ++v) {
        const vec3 u = modes.Shapes[v][mode];
        const vec3 p = modes.Positions[v];
        const double ry = p.y - bar.Width / 2, rz = p.z - bar.Thickness / 2;
        axial += double(u.x) * u.x;
        lateral_y += double(u.y) * u.y;
        lateral_z += double(u.z) * u.z;
        total += numeric::Dot(u, u);
        auto &[circulation, r2] = slices[int(std::lround(p.x * nx / bar.Length))];
        circulation += ry * u.z - rz * u.y;
        r2 += ry * ry + rz * rz;
    }
    if (total <= 0) return Family::Other;

    double rotation = 0; // Energy of the per-slice fitted rigid rotation
    for (const auto &[_, slice] : slices) {
        const auto &[circulation, r2] = slice;
        if (r2 > 0) rotation += circulation * circulation / r2;
    }
    if (axial / total > 0.85) return Family::Longitudinal;
    if (rotation / total > 0.85) return Family::Torsional;
    // Bending carries some axial rotary motion for stubby sections, so the lateral threshold stays loose.
    if (const double lateral = lateral_y + lateral_z; lateral / total > 0.6 && rotation / total < 0.5) {
        if (lateral_y / lateral > 0.8) return Family::BendingY;
        if (lateral_z / lateral > 0.8) return Family::BendingZ;
        return Family::Bending;
    }
    return Family::Other;
}

std::map<Family, std::vector<double>> SolveBar(const Bar &bar, int nx, int ny, int nz) {
    const auto tets = MakeStructuredBar(nx, ny, nz, {bar.Length, bar.Width, bar.Thickness});
    const std::vector<vec3> all_positions(tets.Points.begin(), tets.Points.end());
    const auto result = modal::mesh2modes(tets, bar.Material, all_positions, vec3{1}, {});
    std::map<Family, std::vector<double>> fem;
    for (uint32_t mode = 0; mode < result.Modes.Freqs.size(); ++mode) {
        fem[Classify(result.Modes, mode, bar, nx)].push_back(result.Modes.Freqs[mode]);
    }
    return fem;
}

std::vector<double> HarmonicSeries(double f1) { return {f1, 2 * f1, 3 * f1}; }

std::vector<double> BendingTheory(const Bar &bar, double thickness, int per_root) {
    const double r_gyration = thickness / std::sqrt(12.0);
    const double base = std::sqrt(bar.Material.YoungModulus / bar.Material.Density) * r_gyration / (2 * std::numbers::pi * bar.Length * bar.Length);
    std::vector<double> freqs;
    for (const double bl : BendingBL) {
        for (int i = 0; i < per_root; ++i) freqs.push_back(bl * bl * base);
    }
    return freqs;
}

struct Surface {
    std::vector<dvec3> Points;
    std::vector<uint32_t> Tris;
};

// Axis-aligned box as a k x k grid per face, heavy in exact degeneracies.
// k of 1 is the unit cube.
Surface GridBox(int k) {
    Surface s;
    std::map<std::array<int, 3>, uint32_t> ids;
    const auto vid = [&](int x, int y, int z) {
        const auto [it, inserted] = ids.try_emplace(std::array{x, y, z}, uint32_t(s.Points.size()));
        if (inserted) s.Points.emplace_back(double(x) / k, double(y) / k, double(z) / k);
        return it->second;
    };
    const auto face = [&](auto &&corner) {
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                const uint32_t a = corner(i, j), b = corner(i + 1, j), c = corner(i + 1, j + 1), d = corner(i, j + 1);
                s.Tris.insert(s.Tris.end(), {a, b, c, a, c, d});
            }
        }
    };
    face([&](int i, int j) { return vid(i, j, 0); });
    face([&](int i, int j) { return vid(i, j, k); });
    face([&](int i, int j) { return vid(i, 0, j); });
    face([&](int i, int j) { return vid(i, k, j); });
    face([&](int i, int j) { return vid(0, i, j); });
    face([&](int i, int j) { return vid(k, i, j); });
    return s;
}

Surface Sphere(int subdivisions, double noise, unsigned seed) {
    constexpr double phi = std::numbers::phi;
    std::vector<dvec3> pts{{-1, phi, 0}, {1, phi, 0}, {-1, -phi, 0}, {1, -phi, 0}, {0, -1, phi}, {0, 1, phi}, {0, -1, -phi}, {0, 1, -phi}, {phi, 0, -1}, {phi, 0, 1}, {-phi, 0, -1}, {-phi, 0, 1}};
    std::vector<std::array<uint32_t, 3>> tris{
        {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8}, {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
    };
    for (auto &p : pts) p = numeric::Normalize(p);
    for (int s = 0; s < subdivisions; ++s) {
        std::map<uint64_t, uint32_t> mid;
        const auto midpoint = [&](uint32_t a, uint32_t b) {
            const uint64_t key = (uint64_t(std::min(a, b)) << 32) | std::max(a, b);
            const auto [it, inserted] = mid.try_emplace(key, uint32_t(pts.size()));
            if (inserted) pts.push_back(numeric::Normalize(0.5 * (pts[a] + pts[b])));
            return it->second;
        };
        std::vector<std::array<uint32_t, 3>> next;
        for (const auto &t : tris) {
            const uint32_t ab = midpoint(t[0], t[1]), bc = midpoint(t[1], t[2]), ca = midpoint(t[2], t[0]);
            next.insert(next.end(), {{t[0], ab, ca}, {t[1], bc, ab}, {t[2], ca, bc}, {ab, bc, ca}});
        }
        tris = std::move(next);
    }
    if (noise > 0) {
        std::mt19937 rng{seed};
        std::uniform_real_distribution<double> d{1 - noise, 1 + noise};
        for (auto &p : pts) p *= d(rng);
    }
    Surface s;
    s.Points = std::move(pts);
    for (const auto &t : tris) s.Tris.insert(s.Tris.end(), {t[0], t[1], t[2]});
    return s;
}

// The RealImpact dataset, which REALIMPACT_DATASET_DIR overrides at run time.
std::filesystem::path DatasetDir() {
    const char *env_dataset = std::getenv("REALIMPACT_DATASET_DIR");
    return env_dataset ? env_dataset : REALIMPACT_DATASET_DIR;
}

void CheckFamily(std::string_view name, const std::vector<double> &fem, const std::vector<double> &theory, double tolerance, size_t min_count = 2) {
    const auto count = std::min(fem.size(), theory.size());
    expect(count >= min_count);
    for (size_t i = 0; i < count; ++i) {
        const double ratio = fem[i] / theory[i];
        std::println("{:>12} {}: theory {:8.2f} Hz, FEM {:8.2f} Hz, ratio {:.4f}", name, i + 1, theory[i], fem[i], ratio);
        expect(std::abs(ratio - 1.0) < tolerance);
    }
}

} // namespace

int main() {
    "Tet10 assembly preserves translational mass and rigid modes"_test = [] {
        constexpr AcousticMaterialProperties material{
            .Density = 2700, .YoungModulus = 7.2e10, .PoissonRatio = 0.19, .Alpha = 0, .Beta = 0
        };
        const modal::Tet10Assembler fem{
            {.Points = {{0, 0, 0}, {1.2, 0.1, 0}, {-0.2, 0.9, 0.1}, {0.1, -0.1, 1.1}},
             .Tets = {{0, 1, 2, 3}}},
            material
        };
        const auto [mass, stiffness] = fem.AssembleLower();
        const auto mass_action = mass.selfadjointView<Eigen::Lower>();
        const auto stiffness_action = stiffness.selfadjointView<Eigen::Lower>();
        const double physical_mass = material.Density * fem.Elements().front().Volume;
        for (uint32_t component = 0; component < 3; ++component) {
            Eigen::VectorXd translation = Eigen::VectorXd::Zero(fem.Dofs());
            for (uint32_t node = 0; node < fem.NumNodes; ++node) translation[3 * node + component] = 1;
            expect(std::abs(translation.dot(mass_action * translation) / physical_mass - 1) < 1e-14);
            expect((stiffness_action * translation).norm() < 1e-14 * stiffness.norm() * translation.norm());
        }
    };

    "Accelerate Cholesky reuses symbolic analysis"_test = [] {
        constexpr AcousticMaterialProperties material{.Density = 2700, .YoungModulus = 7.2e10, .PoissonRatio = 0.19};
        const double alpha = std::pow(2 * std::numbers::pi * 20, 2);
        const modal::Tet10Assembler fem{MakeStructuredBar(2, 1, 1), material};
        const auto [mass, stiffness] = fem.AssembleLower();
        const Eigen::SparseMatrix<double> first = stiffness + alpha * mass;
        const Eigen::SparseMatrix<double> second = stiffness + 2 * alpha * mass;
        SparseCholeskySymbolic symbolic{first};
        Eigen::SparseMatrix<double> diagonal(first.rows(), first.cols());
        diagonal.setIdentity();
        expect(symbolic.Matches(second));
        expect(!symbolic.Matches(diagonal));
        SparseCholesky reused{second, symbolic};
        Eigen::MatrixXd rhs(fem.Dofs(), 4), solution(fem.Dofs(), 4);
        for (Eigen::Index column = 0; column < rhs.cols(); ++column)
            for (Eigen::Index row = 0; row < rhs.rows(); ++row) rhs(row, column) = std::cos(0.07 * double((row + 3) * (column + 1)));
        reused.Solve(rhs.data(), solution.data(), rhs.cols());
        expect((second.selfadjointView<Eigen::Lower>() * solution - rhs).norm() / rhs.norm() < 1e-8);
    };

    "block sparse Cholesky reuses symbolic analysis across shifts and solves panels"_test = [] {
        constexpr AcousticMaterialProperties material{.Density = 2700, .YoungModulus = 7.2e10, .PoissonRatio = 0.19};
        const double alpha = std::pow(2 * std::numbers::pi * 20, 2);
        const modal::Tet10Assembler fem{MakeStructuredBar(2, 1, 1), material};
        const auto [mass, stiffness] = fem.AssembleLower();
        BlockSparseCholesky native{fem};
        Eigen::MatrixXd rhs(fem.Dofs(), 4), solution(fem.Dofs(), 4), first_solution, reference(fem.Dofs(), 4);
        for (Eigen::Index column = 0; column < rhs.cols(); ++column)
            for (Eigen::Index row = 0; row < rhs.rows(); ++row) rhs(row, column) = std::sin(0.05 * double((row + 2) * (column + 1)));
        for (const double scale : {1.0, 2.0, 1.0}) {
            native.SetShift(-scale * alpha);
            native.Solve(rhs.data(), solution.data(), rhs.cols());
            const Eigen::SparseMatrix<double> shifted = stiffness + scale * alpha * mass;
            SparseCholesky accelerate{shifted};
            accelerate.Solve(rhs.data(), reference.data(), rhs.cols());
            const double residual = (shifted.selfadjointView<Eigen::Lower>() * solution - rhs).norm() / rhs.norm();
            const double difference = (solution - reference).norm() / reference.norm();
            expect(residual < 1e-8) << residual << solution.norm() << reference.norm();
            expect(difference < 1e-8) << difference << solution.norm() << reference.norm();
            if (first_solution.size() == 0) first_solution = solution;
            else if (scale == 1.0) expect((solution.array() == first_solution.array()).all());
        }
    };

    "native block sparse Tet10 shift-invert certifies the same spectrum"_test = [] {
        constexpr AcousticMaterialProperties material{.Density = 2700, .YoungModulus = 7.2e10, .PoissonRatio = 0.19};
        constexpr int count{12}, basis{36};
        const double shift = std::pow(2 * std::numbers::pi * 20, 2);
        const modal::Tet10Assembler fem{MakeStructuredBar(3, 2, 1), material};
        const auto [mass, stiffness] = fem.AssembleLower();
        double factor_seconds{}, solve_seconds{};
        CholeskyShiftInvert accelerate{stiffness, mass, factor_seconds, solve_seconds};
        BlockSparseCholesky block{fem};
        Spectra::SparseSymMatProd<double> mass_action{mass};
        Spectra::SymGEigsShiftSolver<CholeskyShiftInvert, Spectra::SparseSymMatProd<double>, Spectra::GEigsMode::ShiftInvert> accelerate_solver{
            accelerate, mass_action, count, basis, -shift
        };
        Spectra::SymGEigsShiftSolver<BlockSparseCholesky, Spectra::SparseSymMatProd<double>, Spectra::GEigsMode::ShiftInvert> block_solver{
            block, mass_action, count, basis, -shift
        };
        accelerate_solver.init();
        block_solver.init();
        accelerate_solver.compute(Spectra::SortRule::LargestMagn, 300, 1e-10, Spectra::SortRule::SmallestAlge);
        block_solver.compute(Spectra::SortRule::LargestMagn, 300, 1e-10, Spectra::SortRule::SmallestAlge);
        expect(accelerate_solver.info() == Spectra::CompInfo::Successful);
        expect(block_solver.info() == Spectra::CompInfo::Successful);
        const Eigen::VectorXd reference = accelerate_solver.eigenvalues();
        const Eigen::VectorXd values = block_solver.eigenvalues();
        expect((values - reference).norm() / reference.norm() < 1e-9);
        const Eigen::MatrixXd vectors = block_solver.eigenvectors();
        const auto k = stiffness.selfadjointView<Eigen::Lower>();
        const auto m = mass.selfadjointView<Eigen::Lower>();
        double maximum_residual{};
        for (int mode = 6; mode < count; ++mode) {
            const Eigen::VectorXd kx = k * vectors.col(mode), mx = m * vectors.col(mode);
            maximum_residual = std::max(maximum_residual, (kx - values[mode] * mx).norm() / (kx.norm() + std::abs(values[mode]) * mx.norm()));
        }
        expect(maximum_residual < 1e-8) << maximum_residual;
    };

    "block sparse Cholesky reassembles fixed topology and rescales material without new symbolic analysis"_test = [] {
        constexpr AcousticMaterialProperties material{.Density = 2700, .YoungModulus = 7.2e10, .PoissonRatio = 0.19};
        TetMesh mesh = MakeStructuredBar(2, 1, 1);
        const modal::Tet10Assembler original{mesh, material};
        BlockSparseCholesky factor{original};
        for (auto &point : mesh.Points) point.x *= 1.07 + 0.03 * point.y / 0.05;
        const modal::Tet10Assembler deformed{mesh, material};
        factor.Reassemble(deformed);
        constexpr double stiffness_scale{1.2}, mass_scale{0.8};
        factor.ScalePencil(stiffness_scale, mass_scale);
        const AcousticMaterialProperties scaled_material{
            .Density = material.Density * mass_scale,
            .YoungModulus = material.YoungModulus * stiffness_scale,
            .PoissonRatio = material.PoissonRatio,
        };
        const modal::Tet10Assembler scaled{mesh, scaled_material};
        const auto [mass, stiffness] = scaled.AssembleLower();
        const double shift = std::pow(2 * std::numbers::pi * 20, 2);
        const Eigen::SparseMatrix<double> shifted = stiffness + shift * mass;
        Eigen::MatrixXd rhs(scaled.Dofs(), 4), solution(scaled.Dofs(), 4), reference(scaled.Dofs(), 4);
        rhs.setRandom();
        factor.SetShift(-shift);
        factor.Solve(rhs.data(), solution.data(), int(rhs.cols()));
        SparseCholesky accelerate{shifted};
        accelerate.Solve(rhs.data(), reference.data(), int(rhs.cols()));
        const double residual_norm = (shifted.selfadjointView<Eigen::Lower>() * solution - rhs).norm();
        const double residual = residual_norm / rhs.norm();
        const double backward_error = residual_norm / (shifted.norm() * solution.norm() + rhs.norm());
        const double difference = (solution - reference).norm() / reference.norm();
        expect(residual < 2e-8) << residual;
        expect(backward_error < 1e-12) << backward_error;
        expect(difference < 2e-8) << difference;
    };
    "solve cache reuses Tet10 topology and assembly across material edits"_test = [] {
        constexpr AcousticMaterialProperties material{.Density = 2700, .YoungModulus = 7.2e10, .PoissonRatio = 0.19};
        const TetMesh mesh = MakeStructuredBar(2, 1, 1);
        modal::SolveCache cache;
        constexpr modal::SolverConfig config{.NumModes = 12, .NumFemModes = 12, .Tolerance = 1e-8};
        const auto first = modal::mesh2modes(mesh, material, {vec3(mesh.Points.front())}, vec3{1}, config, {.Cache = &cache, .KeepBasis = true});
        const auto second = modal::mesh2modes(mesh, material, {vec3(mesh.Points.front())}, vec3{1}, config, {.Cache = &cache, .KeepBasis = true});
        auto changed_material = material;
        changed_material.YoungModulus *= 1.5;
        const auto changed = modal::mesh2modes(mesh, changed_material, {vec3(mesh.Points.front())}, vec3{1}, config, {.SeedBasis = &first.Basis, .Cache = &cache, .KeepBasis = true});
        TetMesh deformed_mesh = mesh;
        for (auto &point : deformed_mesh.Points) point.x *= 1.03 + point.y;
        const auto deformed = modal::mesh2modes(deformed_mesh, changed_material, {vec3(deformed_mesh.Points.front())}, vec3{1}, config, {.SeedBasis = &changed.Basis, .Cache = &cache, .KeepBasis = true});
        const auto deformed_cold = modal::mesh2modes(deformed_mesh, changed_material, {vec3(deformed_mesh.Points.front())}, vec3{1}, config, {.KeepBasis = true});
        expect(!first.Profile.TopologyReuse);
        expect(!first.Profile.AssemblyReuse);
        expect(second.Profile.TopologyReuse);
        expect(second.Profile.AssemblyReuse);
        expect(second.Profile.SymbolicReuse);
        expect(changed.Profile.TopologyReuse);
        expect(changed.Profile.AssemblyReuse);
        expect(changed.Profile.SymbolicReuse);
        expect(!deformed.Profile.TopologyReuse);
        expect(!deformed.Profile.AssemblyReuse);
        expect(deformed.Profile.SymbolicReuse);
        expect(first.Summary.Eigenvalues.size() == 12);
        expect(second.Summary.Eigenvalues.size() == 12);
        expect(changed.Summary.Eigenvalues.size() == 12);
        expect(deformed.Summary.Eigenvalues.size() == 12);
        expect(deformed_cold.Summary.Eigenvalues.size() == 12);
        expect(first.Summary.Eigenvalues == second.Summary.Eigenvalues);
        expect(bool((first.Basis.array() == second.Basis.array()).all()));
        for (uint32_t mode = 6; mode < 12; ++mode) {
            expect(std::abs(second.Summary.Eigenvalues[mode] / first.Summary.Eigenvalues[mode] - 1) < 1e-8);
            expect(std::abs(changed.Summary.Eigenvalues[mode] / first.Summary.Eigenvalues[mode] - 1.5) < 1e-8);
        }
        expect(deformed.Profile.PhysicalResidual < 1e-8) << deformed.Profile.PhysicalResidual;
        expect(deformed.Profile.MassOrthogonality < 1e-8);
        for (uint32_t mode = 6; mode < 12; ++mode)
            expect(std::abs(deformed.Summary.Eigenvalues[mode] / deformed_cold.Summary.Eigenvalues[mode] - 1) < 1e-8);
    };

    // Square section: longitudinal validates the E/rho/assembly/eigensolve chain end to end,
    // torsion and bending validate shear response.
    "square bar modes match closed forms"_test = [] {
        const Bar bar{.Length = 0.3, .Width = 0.05, .Thickness = 0.05, .Material = {.Density = 1000, .YoungModulus = 1e7, .PoissonRatio = 0, .Alpha = 0, .Beta = 0}};
        const double speed = std::sqrt(bar.Material.YoungModulus / bar.Material.Density);
        constexpr double TorsionOverPolar = 0.140577 * 6; // J/Ip for a square section
        const double torsion_f1 = std::sqrt(bar.Material.Mu() / bar.Material.Density * TorsionOverPolar) / (2 * bar.Length);
        std::println("--- square bar 20x4x4 ---");
        auto fem = SolveBar(bar, 20, 4, 4);
        // Degenerate pairs may mix planes, so pool all bending buckets.
        auto bending = fem[Family::Bending];
        for (const auto f : {Family::BendingY, Family::BendingZ}) bending.insert(bending.end(), fem[f].begin(), fem[f].end());
        std::ranges::sort(bending);
        CheckFamily("longitudinal", fem[Family::Longitudinal], HarmonicSeries(speed / (2 * bar.Length)), 0.01);
        CheckFamily("torsional", fem[Family::Torsional], HarmonicSeries(torsion_f1), 0.05);
        // Euler-Bernoulli overestimates higher modes of a stubby bar (no shear/rotary inertia), so
        // compare only the first degenerate pair.
        bending.resize(std::min<size_t>(bending.size(), 2));
        CheckFamily("bending", bending, BendingTheory(bar, bar.Thickness, 2), 0.10);
    };

    // Thin section with a single element through the thickness, matching how thin-walled objects
    // tetrahedralize in practice. Quadratic elements capture the bending strain through one element.
    "thin bar bending matches closed forms"_test = [] {
        const Bar bar{.Length = 0.3, .Width = 0.05, .Thickness = 0.01, .Material = {.Density = 1000, .YoungModulus = 1e9, .PoissonRatio = 0, .Alpha = 0, .Beta = 0}};
        const double speed = std::sqrt(bar.Material.YoungModulus / bar.Material.Density);
        std::println("--- thin bar 30x5x1 ---");
        auto fem = SolveBar(bar, 30, 5, 1);
        CheckFamily("longitudinal", fem[Family::Longitudinal], HarmonicSeries(speed / (2 * bar.Length)), 0.01);
        // Euler-Bernoulli overestimates the stiff plane's higher modes (no shear/rotary inertia), so
        // compare only its first mode.
        auto bending_y_theory = BendingTheory(bar, bar.Width, 1);
        bending_y_theory.resize(1);
        CheckFamily("bending-y", fem[Family::BendingY], bending_y_theory, 0.10, 1);
        CheckFamily("bending-z", fem[Family::BendingZ], BendingTheory(bar, bar.Thickness, 1), 0.05);
    };

    // Exact vertex coordinates put the predicates on their degenerate cases: coplanar faces, cospherical corners, collinear edges.
    // Noise on the sphere moves them to near-degenerate instead.
    "synthetic shapes tetrahedralize to valid meshes"_test = [] {
        const std::pair<std::string_view, Surface> cases[]{
            {"cube", GridBox(1)},
            {"grid box 4", GridBox(4)},
            {"grid box 7", GridBox(7)},
            {"sphere", Sphere(3, 0, 0)},
            {"noisy sphere", Sphere(3, 0.05, 7)},
        };
        for (const auto &[name, surface] : cases) {
            for (const bool quality : {false, true}) {
                const auto tets = tetra::Tetrahedralize(surface.Points, surface.Tris, {.Quality = quality});
                if (!tets) {
                    expect(false) << name << "tetrahedralization failed:" << tets.error();
                    continue;
                }
                const auto err = ValidateTetMesh(surface.Points, surface.Tris, tets->Mesh);
                expect(err.empty()) << name << "invalid tet mesh:" << err;
            }
        }
    };

    "tetrahedralizer cavity seeds remove enclosed voids"_test = [] {
        Surface surface = GridBox(2), inner = GridBox(2);
        for (dvec3 &point : inner.Points) point = 0.25 + 0.5 * point;
        const uint32_t offset = uint32_t(surface.Points.size());
        surface.Points.insert(surface.Points.end(), inner.Points.begin(), inner.Points.end());
        for (const uint32_t point : inner.Tris) surface.Tris.push_back(offset + point);
        const std::array holes{dvec3{0.5}};
        const auto result = tetra::Tetrahedralize(surface.Points, surface.Tris, {.Quality = true, .Holes = holes});
        expect(bool(result));
        if (!result) return;
        expect(ValidateTetMesh(surface.Points, surface.Tris, result->Mesh).empty());
        double volume{};
        for (const auto &tet : result->Mesh.Tets) {
            const dvec3 center = 0.25 * (result->Mesh.Points[tet[0]] + result->Mesh.Points[tet[1]] + result->Mesh.Points[tet[2]] + result->Mesh.Points[tet[3]]);
            expect(center.x < 0.25 || center.y < 0.25 || center.z < 0.25 || center.x > 0.75 || center.y > 0.75 || center.z > 0.75);
            volume += std::abs(geom::Orient3D(result->Mesh.Points[tet[0]], result->Mesh.Points[tet[1]], result->Mesh.Points[tet[2]], result->Mesh.Points[tet[3]])) / 6;
        }
        expect(std::abs(volume - 0.875) < 1e-12);
    };

    "RealImpact bowl solves in reasonable time"_test = [] {
        const auto path = DatasetDir() / "9_BowlCeramic/preprocessed/transformed.obj";
        if (!std::filesystem::exists(path)) {
            std::println("skipping RealImpact benchmark: {} not found", path.string());
            return;
        }
        auto surface = LoadObj(path);
        if (!surface || surface->Positions.empty()) {
            std::println("skipping RealImpact benchmark: no mesh data in {}", path.string());
            return;
        }
        const std::vector<vec3> excite{surface->Positions.front()};
        const auto n_verts = surface->Positions.size(), n_tris = surface->TriangleIndices.size() / 3;
        const auto tets = GenerateTets(std::move(surface->Positions), std::move(surface->TriangleIndices), {});
        expect(tets.has_value());
        if (!tets) return;
        std::println("--- RealImpact bowl: {} welded verts, {} tris -> {} tet nodes, {} tets ---", n_verts, n_tris, tets->Mesh.Points.size(), tets->Mesh.Tets.size());

        constexpr AcousticMaterialProperties Ceramic{.Density = 2700, .YoungModulus = 7.2e10, .PoissonRatio = 0.19, .Alpha = 5, .Beta = 1e-8};
        const auto start = std::chrono::steady_clock::now();
        const auto result = modal::mesh2modes(tets->Mesh, Ceramic, excite, vec3{1}, {});
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        expect(!result.Modes.Freqs.empty());
        std::println("{:6.2f} s, {} modes, f1 {:8.1f} Hz", seconds, result.Modes.Freqs.size(), result.Modes.Freqs.empty() ? 0.0 : double(result.Modes.Freqs.front()));
    };

    // Every RealImpact object must tetrahedralize to a structurally valid mesh at every
    // resolution the app's simplification slider offers: simplifying a watertight, non-self-
    // intersecting surface should keep it tetrahedralizable. Skipped when the dataset is absent.
    "RealImpact meshes tetrahedralize to valid meshes across resolutions"_test = [] {
        const auto dataset = DatasetDir();
        if (!std::filesystem::exists(dataset)) {
            std::println("skipping RealImpact validation: {} not found", dataset.string());
            return;
        }
        std::vector<std::filesystem::path> objs;
        for (const auto &entry : std::filesystem::directory_iterator{dataset})
            if (entry.is_directory() && std::filesystem::exists(entry.path() / "preprocessed" / "transformed.obj"))
                objs.push_back(entry.path() / "preprocessed" / "transformed.obj");
        std::ranges::sort(objs);
        expect(!objs.empty());
        for (const auto &path : objs) {
            const auto name = path.parent_path().parent_path().filename().string();
            const auto surface = LoadObj(path);
            if (!surface || surface->Positions.empty()) {
                expect(false) << name << "failed to load";
                continue;
            }
            for (const float ratio : {1.0f, 0.5f, 0.25f}) {
                auto positions = surface->Positions;
                auto triangle_indices = surface->TriangleIndices;
                SimplifySurface(positions, triangle_indices, ratio);
                const std::vector<dvec3> in_points(positions.begin(), positions.end());
                const auto tets = GenerateTets(positions, triangle_indices, {});
                if (!tets) {
                    expect(false) << name << "@" << ratio << "tetrahedralization failed:" << tets.error();
                    continue;
                }
                const auto err = ValidateTetMesh(in_points, triangle_indices, tets->Mesh);
                expect(err.empty()) << name << "@" << ratio << "invalid tet mesh:" << err;
            }
        }
    };

    return RunSuites();
}
