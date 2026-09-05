#include "LoadObj.h"
#include <FastFEM/Surface2Modes.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <numbers>
#include <print>
#include <ranges>
#include <stdexcept>
#include <string_view>

namespace {
const char *ArgValue(int argc, char **argv, std::string_view name, const char *fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (argv[i] == name) return argv[i + 1];
    return fallback;
}

double ArgValue(int argc, char **argv, std::string_view name, double fallback) {
    const auto text = ArgValue(argc, argv, name, nullptr);
    if (!text) return fallback;
    const std::string_view s = text;
    double value;
    const auto [end, error] = std::from_chars(s.data(), s.data() + s.size(), value);
    if (error == std::errc{} && end == s.data() + s.size() && std::isfinite(value)) return value;
    throw std::invalid_argument("Invalid value for " + std::string{name});
}

uint32_t Count(int argc, char **argv, std::string_view name, uint32_t fallback, bool allow_zero = false) {
    const double value = ArgValue(argc, argv, name, double(fallback));
    if (value < (allow_zero ? 0 : 1) || value > UINT32_MAX || std::floor(value) != value)
        throw std::invalid_argument("Invalid count for " + std::string{name});
    return uint32_t(value);
}

void PrintArray(std::string_view key, auto &&values) {
    std::print("  \"{}\": [", key);
    std::string_view separator;
    for (const auto value : values) {
        if constexpr (requires { value.x; }) std::print("{}[{},{},{}]", separator, value.x, value.y, value.z);
        else std::print("{}{}", separator, value);
        separator = ",";
    }
    std::println("],");
}
} // namespace

int main(int argc, char **argv) try {
    if (argc < 2) {
        std::println(stderr, "Usage: {} <mesh.obj> [--discretization tet10|finite-cell] [--young E] [--poisson v] [--density rho] [--alpha a] [--beta b] [--min-freq f] [--max-freq f] [--modes n] [--fem-modes n] [--tolerance r] [--restarts n] [--refinement none|quality|quality-and-resolution] [--resolution n] [--cut-depth n] [--benchmark]", argv[0]);
        return 1;
    }
    const auto mesh = LoadObj(argv[1]);
    if (!mesh || mesh->Positions.empty()) {
        std::println(stderr, "Failed to load mesh: {}", argv[1]);
        return 1;
    }

    bool benchmark{};
    for (int i = 2; i < argc; ++i) {
        const std::string_view name = argv[i];
        if (name == "--benchmark") {
            benchmark = true;
            continue;
        }
        constexpr std::array options{"--discretization", "--young", "--poisson", "--density", "--alpha", "--beta", "--min-freq", "--max-freq", "--modes", "--fem-modes", "--tolerance", "--restarts", "--refinement", "--resolution", "--cut-depth"};
        if (std::ranges::find(options, name) == options.end() || i + 1 == argc)
            throw std::invalid_argument("Unknown option or missing value: " + std::string{name});
        ++i;
    }
    const fastfem::AcousticMaterialProperties material{
        .Density = ArgValue(argc, argv, "--density", 2700),
        .YoungModulus = ArgValue(argc, argv, "--young", 7.2e10),
        .PoissonRatio = ArgValue(argc, argv, "--poisson", 0.19),
        .Alpha = ArgValue(argc, argv, "--alpha", 5),
        .Beta = ArgValue(argc, argv, "--beta", 2e-8),
    };
    const std::string_view discretization_name = ArgValue(argc, argv, "--discretization", "tet10");
    fastfem::Discretization discretization;
    if (discretization_name == "tet10") discretization = fastfem::Discretization::Tet10;
    else if (discretization_name == "finite-cell") discretization = fastfem::Discretization::FiniteCell;
    else throw std::invalid_argument("Unknown discretization: " + std::string{discretization_name});
    const std::string_view refinement_name = ArgValue(argc, argv, "--refinement", "none");
    fastfem::TetRefinement refinement;
    if (refinement_name == "none") refinement = fastfem::TetRefinement::None;
    else if (refinement_name == "quality") refinement = fastfem::TetRefinement::Quality;
    else if (refinement_name == "quality-and-resolution") refinement = fastfem::TetRefinement::QualityAndResolution;
    else throw std::invalid_argument("Unknown refinement: " + std::string{refinement_name});
    const uint32_t modes_requested = Count(argc, argv, "--modes", 30);
    const fastfem::SurfaceSolveConfig config{
        .Modal = {
            .MinModeFreq = float(ArgValue(argc, argv, "--min-freq", 20)),
            .MaxModeFreq = float(ArgValue(argc, argv, "--max-freq", 16'000)),
            .NumModes = modes_requested,
            .NumFemModes = Count(argc, argv, "--fem-modes", modes_requested + 15),
            .Tolerance = ArgValue(argc, argv, "--tolerance", 1e-8),
            .MaxRestarts = Count(argc, argv, "--restarts", 100),
        },
        .Tetrahedralization = {.Refinement = refinement},
        .FiniteCell = {.CutDepth = Count(argc, argv, "--cut-depth", 3, true)},
        .Resolution = Count(argc, argv, "--resolution", 12),
    };

    std::vector<fastfem::Vec3> samples;
    if (benchmark) {
        const size_t count = std::min(size_t{64}, mesh->Positions.size());
        for (size_t i = 0; i < count; ++i) samples.push_back(mesh->Positions[count == 1 ? 0 : i * (mesh->Positions.size() - 1) / (count - 1)]);
    }
    const auto start = std::chrono::steady_clock::now();
    const auto result = fastfem::Surface2Modes(mesh->Positions, mesh->TriangleIndices, material, benchmark ? samples : mesh->Positions, {1, 1, 1}, discretization, config);
    const double solve_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (!result) {
        std::println(stderr, "Modal solve failed: {}", result.error());
        return 1;
    }
    if (benchmark) {
        if (result->Summary.Eigenvalues.size() != config.Modal.NumFemModes || config.Modal.NumFemModes <= 6)
            throw std::runtime_error("Benchmark requires the full requested spectrum, including physical modes.");
        std::println("{{\"solve_seconds\":{},\"input_vertices\":{},\"input_triangles\":{},\"tetrahedra\":{},\"mass\":{},", solve_seconds, mesh->Positions.size(), mesh->TriangleIndices.size() / 3, result->Tetrahedra.Tets.size(), result->Mass.Mass);
        PrintArray("frequencies", result->Summary.Eigenvalues | std::views::drop(6) | std::views::transform([](double eigenvalue) {
                                      const double frequency = std::sqrt(eigenvalue) / (2 * std::numbers::pi);
                                      if (!std::isfinite(frequency) || frequency <= 0) throw std::runtime_error("Invalid physical eigenfrequency.");
                                      return frequency;
                                  }));
        PrintArray("samples", samples);
        std::print("\"shapes\":[");
        for (size_t mode = 6; mode < result->Summary.Eigenvalues.size(); ++mode) {
            std::print("{}[", mode == 6 ? "" : ",");
            for (size_t i = 0; i < samples.size(); ++i) {
                const auto &s = result->Summary.Shapes[result->SamplePointOfExcitation[i]][mode];
                std::print("{}{},{},{}", i ? "," : "", s.x, s.y, s.z);
            }
            std::print("]");
        }
        std::println("]}}");
        return 0;
    }
    const auto &modes = result->Modes;
    if (modes.Freqs.empty()) {
        std::println(stderr, "Solve produced no modes in [{} Hz, {} Hz]", config.Modal.MinModeFreq, config.Modal.MaxModeFreq);
        return 1;
    }

    // The mesh's triangles, relabeled onto the sample points its vertices became.
    // A triangle whose corners merged onto fewer than three points has no area and is dropped.
    std::vector<uint32_t> indices;
    indices.reserve(mesh->TriangleIndices.size());
    for (size_t t = 0; t + 2 < mesh->TriangleIndices.size(); t += 3) {
        const auto a = result->SamplePointOfExcitation[mesh->TriangleIndices[t]];
        const auto b = result->SamplePointOfExcitation[mesh->TriangleIndices[t + 1]];
        const auto c = result->SamplePointOfExcitation[mesh->TriangleIndices[t + 2]];
        if (a == b || b == c || a == c) continue;
        indices.insert(indices.end(), {a, b, c});
    }

    static constexpr float Ln1000 = 3 * std::numbers::ln10_v<float>; // Converts T60 to exponential decay rate.
    std::vector<float> decay_rates(modes.T60s.size());
    for (size_t k = 0; k < modes.T60s.size(); ++k) decay_rates[k] = modes.T60s[k] > 0 ? Ln1000 / modes.T60s[k] : 0.f;

    std::println("{{");
    PrintArray("frequencies", modes.Freqs);
    PrintArray("decayRates", decay_rates);
    PrintArray("positions", modes.Positions);
    // Stores every sample point for each successive mode to match the model schema.
    std::print("  \"shapes\": [");
    for (size_t k = 0; k < modes.Freqs.size(); ++k) {
        for (size_t i = 0; i < modes.Shapes.size(); ++i) {
            const auto &s = modes.Shapes[i][k];
            std::print("{}[{},{},{}]", k || i ? "," : "", s.x, s.y, s.z);
        }
    }
    std::println("],");
    PrintArray("indices", indices);
    std::println("  \"mass\": {},", result->Mass.Mass);
    std::println("  \"centerOfMass\": [{},{},{}],", result->Mass.CenterOfMass.x, result->Mass.CenterOfMass.y, result->Mass.CenterOfMass.z);
    std::println("  \"inertiaDiagonal\": [{},{},{}]", result->Mass.InertiaDiagonal.x, result->Mass.InertiaDiagonal.y, result->Mass.InertiaDiagonal.z);
    std::println("}}");
    return 0;
} catch (const std::exception &error) {
    std::println(stderr, "{}", error.what());
    return 1;
}
