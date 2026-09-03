#include "LoadObj.h"
#include "StructuredBar.h"
#include "audio/BlockSparseCholesky.h"
#include "audio/SparseCholesky.h"
#include "audio/Tet10Assembler.h"
#include "mesh/Tets.h"

#include <Eigen/Core>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

double Median(std::vector<double> values) {
    std::ranges::sort(values);
    const size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : 0.5 * (values[middle - 1] + values[middle]);
}

uint64_t Hash(const Eigen::MatrixXd &matrix) {
    uint64_t hash{1469598103934665603ull};
    for (const double value : std::span{matrix.data(), size_t(matrix.size())}) {
        hash ^= std::bit_cast<uint64_t>(value);
        hash *= 1099511628211ull;
    }
    return hash;
}

size_t PeakResidentBytes() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage)) return 0;
    return size_t(usage.ru_maxrss);
}

struct Result {
    double Setup{}, Numeric{}, Solve{}, Residual{};
    uint64_t SolutionHash{};
    bool Deterministic{};
};

struct AccelerateState {
    SparseCholeskySymbolic Symbolic;
    std::unique_ptr<SparseCholesky> Factor;

    explicit AccelerateState(const Eigen::SparseMatrix<double> &matrix) : Symbolic(matrix) {}
};

template<class Setup, class Numeric, class Solve>
Result Measure(
    int repetitions, Setup &&setup, Numeric &&numeric, Solve &&solve,
    const std::array<Eigen::SparseMatrix<double>, 2> &shifted, const Eigen::MatrixXd &rhs
) {
    const auto setup_start = Clock::now();
    auto factor = setup();
    Result result{.Setup = Seconds(setup_start)};
    std::vector<double> numeric_seconds, solve_seconds;
    Eigen::MatrixXd solution(rhs.rows(), rhs.cols());
    std::array<Eigen::MatrixXd, 2> first;
    std::array<bool, 2> seen{};
    numeric_seconds.reserve(repetitions);
    solve_seconds.reserve(repetitions);
    result.Deterministic = true;
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        const size_t shift = size_t(repetition) % shifted.size();
        auto start = Clock::now();
        numeric(factor, shift);
        numeric_seconds.push_back(Seconds(start));
        start = Clock::now();
        solve(factor, solution);
        solve_seconds.push_back(Seconds(start));
        if (!seen[shift]) {
            first[shift] = solution;
            seen[shift] = true;
        } else {
            result.Deterministic &= (solution.array() == first[shift].array()).all();
        }
    }
    result.Numeric = Median(std::move(numeric_seconds));
    result.Solve = Median(std::move(solve_seconds));
    const auto &final_shifted = shifted[size_t(repetitions - 1) % shifted.size()];
    result.Residual = (final_shifted.selfadjointView<Eigen::Lower>() * solution - rhs).norm() / rhs.norm();
    result.SolutionHash = Hash(solution);
    return result;
}

void Print(std::string_view route, const Result &result) {
    std::println("{} setup={:.6f}s numeric={:.6f}s solve={:.6f}s residual={:.3e} deterministic={} hash={:016x}", route, result.Setup, result.Numeric, result.Solve, result.Residual, result.Deterministic, result.SolutionHash);
}

template<class NativeSetup>
void BenchmarkPencil(
    std::string_view name, const Eigen::SparseMatrix<double> &mass, const Eigen::SparseMatrix<double> &stiffness,
    int repetitions, int width, std::string_view backend, NativeSetup &&native_setup
) {
    const double shift = std::pow(2 * std::numbers::pi * 20, 2);
    const std::array shifts{shift, 1.001 * shift};
    const std::array<Eigen::SparseMatrix<double>, 2> shifted{
        stiffness + shifts[0] * mass,
        stiffness + shifts[1] * mass,
    };
    Eigen::MatrixXd rhs(stiffness.rows(), width);
    for (Eigen::Index column = 0; column < rhs.cols(); ++column)
        for (Eigen::Index row = 0; row < rhs.rows(); ++row)
            rhs(row, column) = std::sin(0.013 * double(row + 1) + 0.17 * double(column + 1));

    std::println("{} dofs={} matrix_nnz={}", name, stiffness.rows(), shifted.front().nonZeros());
    if (backend == "all" || backend == "accelerate") {
        Print("accelerate", Measure(repetitions, [&] { return AccelerateState{shifted.front()}; }, [&](AccelerateState &state, size_t index) { state.Factor = std::make_unique<SparseCholesky>(shifted[index], state.Symbolic); }, [&](const AccelerateState &state, Eigen::MatrixXd &solution) { state.Factor->Solve(rhs.data(), solution.data(), width); }, shifted, rhs));
    }
    if (backend == "all" || backend == "native") {
        Print("native", Measure(repetitions, [&] { return native_setup(); }, [&](BlockSparseCholesky &factor, size_t index) { factor.SetShift(-shifts[index]); }, [&](const BlockSparseCholesky &factor, Eigen::MatrixXd &solution) { factor.Solve(rhs.data(), solution.data(), width); }, shifted, rhs));
    }
    std::println("peak_resident={}B", PeakResidentBytes());
}

int Integer(const char *text, std::string_view name) {
    const int value = std::stoi(text);
    if (value < 1) throw std::invalid_argument(std::string{name} + " must be positive");
    return value;
}

void ValidateBackend(std::string_view backend) {
    if (backend == "all" || backend == "accelerate" || backend == "native") return;
    throw std::invalid_argument("unsupported backend for this benchmark route");
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc < 2) {
            std::println(stderr, "usage: {} --tet NX NY NZ [REPETITIONS] [all|accelerate|native] [WIDTH] | --obj PATH [REPETITIONS] [all|accelerate|native] [WIDTH]", argv[0]);
            return 2;
        }
        constexpr AcousticMaterialProperties material{.Density = 2700, .YoungModulus = 7.2e10, .PoissonRatio = 0.19};
        if (std::string_view{argv[1]} == "--tet") {
            if (argc < 5) throw std::invalid_argument("--tet requires NX NY NZ");
            const int nx = Integer(argv[2], "NX"), ny = Integer(argv[3], "NY"), nz = Integer(argv[4], "NZ");
            const int repetitions = argc > 5 ? Integer(argv[5], "REPETITIONS") : 5;
            const std::string_view backend = argc > 6 ? argv[6] : "all";
            const int width = argc > 7 ? Integer(argv[7], "WIDTH") : 16;
            ValidateBackend(backend);
            const modal::Tet10Assembler fem{MakeStructuredBar(nx, ny, nz), material};
            const auto assembly_start = Clock::now();
            const auto [mass, stiffness] = fem.AssembleLower();
            std::println("tet10 assembly={:.6f}s elements={}", Seconds(assembly_start), fem.Elements().size());
            BenchmarkPencil("tet10", mass, stiffness, repetitions, width, backend, [&] { return BlockSparseCholesky{fem}; });
            return 0;
        }
        if (std::string_view{argv[1]} == "--obj") {
            if (argc < 3) throw std::invalid_argument("--obj requires PATH");
            const int repetitions = argc > 3 ? Integer(argv[3], "REPETITIONS") : 5;
            const std::string_view backend = argc > 4 ? argv[4] : "all";
            const int width = argc > 5 ? Integer(argv[5], "WIDTH") : 16;
            ValidateBackend(backend);
            const auto surface = LoadObj(argv[2]);
            if (!surface) throw std::runtime_error("failed to load OBJ");
            const auto tets = GenerateTets(surface->Positions, surface->TriangleIndices);
            if (!tets) throw std::runtime_error("tetrahedralization failed: " + tets.error());
            const modal::Tet10Assembler fem{tets->Mesh, material};
            const auto assembly_start = Clock::now();
            const auto [mass, stiffness] = fem.AssembleLower();
            std::println("obj assembly={:.6f}s elements={}", Seconds(assembly_start), fem.Elements().size());
            BenchmarkPencil("obj", mass, stiffness, repetitions, width, backend, [&] { return BlockSparseCholesky{fem}; });
            return 0;
        }
        throw std::invalid_argument("unknown benchmark route");
    } catch (const std::exception &error) {
        std::println(stderr, "error: {}", error.what());
        return 1;
    }
}
