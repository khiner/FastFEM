#include "ModalResultBuilder.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <span>

namespace {
using uint = uint32_t;

ModalModes PostprocessModes(
    std::span<const double> eigenvalues, const std::vector<std::vector<vec3>> &shapes,
    float shape_scale, const AcousticMaterialProperties &material, const modal::SolverConfig &config,
    std::vector<vec3> positions
) {
    const uint fem_n_modes = eigenvalues.size();
    std::vector<float> mode_freqs(fem_n_modes), mode_t60s(fem_n_modes);
    std::vector<double> omega_undamped(fem_n_modes);
    // The eigensolver shift scales the near-zero eigenvalue cutoff.
    const double shift_omega = 2 * std::numbers::pi * config.MinModeFreq;
    const double lambda_eps = shift_omega * shift_omega * 1e-10;
    for (uint mode = 0; mode < fem_n_modes; ++mode) {
        const double lambda_i = eigenvalues[mode];
        omega_undamped[mode] = lambda_i > lambda_eps ? std::sqrt(lambda_i) : 0;
    }

    const auto c_from_omega = [&material](double omega) { return material.Alpha + material.Beta * (omega * omega); };
    const auto damped_hz = [&](double omega, double c) {
        const double omega_d_sq = omega * omega - 0.25 * c * c;
        return omega_d_sq > 0 ? std::sqrt(omega_d_sq) / (2 * std::numbers::pi) : 0;
    };

    uint lowest_mode_i = fem_n_modes;
    float lowest_mode_freq_orig{0};
    for (uint mode = 0; mode < fem_n_modes; ++mode) {
        const double omega_i = omega_undamped[mode];
        if (omega_i <= 0) {
            mode_freqs[mode] = mode_t60s[mode] = 0.f;
            continue;
        }
        mode_freqs[mode] = damped_hz(omega_i, c_from_omega(omega_i));
        // The audible-frequency floor excludes numerically nonzero rigid-body modes.
        if (lowest_mode_i == fem_n_modes && mode_freqs[mode] >= config.MinModeFreq) {
            lowest_mode_i = mode;
            lowest_mode_freq_orig = mode_freqs[mode];
        }
    }
    if (lowest_mode_i == fem_n_modes) return {};

    static const double ln_1000 = std::log(1000);
    const float freq_scale = config.FundamentalFreq ? *config.FundamentalFreq / lowest_mode_freq_orig : 1.f;
    for (uint mode = lowest_mode_i; mode < fem_n_modes; ++mode) {
        const double omega_s = omega_undamped[mode] * freq_scale; // scaled rad/s
        const double c = c_from_omega(omega_s);
        mode_freqs[mode] = damped_hz(omega_s, c);
        mode_t60s[mode] = c > 0 ? (2 * ln_1000) / c : 0;
    }
    // Applying the upper frequency cutoff before fundamental scaling preserves higher modes after retuning.
    const float max_mode_freq = config.MaxModeFreq * std::max(1.f, freq_scale);
    uint highest_mode_i = fem_n_modes;
    while (highest_mode_i > lowest_mode_i && mode_freqs[highest_mode_i - 1] > max_mode_freq) --highest_mode_i;

    const uint n_modes = std::min({config.NumModes, fem_n_modes, highest_mode_i - lowest_mode_i});
    mode_freqs.erase(mode_freqs.begin(), mode_freqs.begin() + lowest_mode_i);
    mode_freqs.resize(n_modes);
    mode_t60s.erase(mode_t60s.begin(), mode_t60s.begin() + lowest_mode_i);
    mode_t60s.resize(n_modes);

    std::vector<std::vector<vec3>> out_shapes(shapes.size(), std::vector<vec3>(n_modes));
    for (size_t ex_pos = 0; ex_pos < shapes.size(); ++ex_pos) {
        for (uint mode = 0; mode < n_modes; ++mode)
            out_shapes[ex_pos][mode] = shapes[ex_pos][mode + lowest_mode_i] * shape_scale;
    }

    return {
        .Freqs = std::move(mode_freqs),
        .T60s = std::move(mode_t60s),
        .Shapes = std::move(out_shapes),
        .Positions = std::move(positions),
        .OriginalFundamentalFreq = lowest_mode_freq_orig,
    };
}
} // namespace

std::optional<ModalModes> modal::RescaleModes(
    const ModalEigenSummary &summary, const ModalModes &current,
    const AcousticMaterialProperties &material, SolverConfig config
) {
    if (summary.Eigenvalues.empty() || material.PoissonRatio != summary.SolvedMaterial.PoissonRatio) return {};

    const double rho_ratio = material.Density / summary.SolvedMaterial.Density;
    const double eigenvalue_scale = (material.YoungModulus / summary.SolvedMaterial.YoungModulus) / rho_ratio;
    auto eigenvalues = summary.Eigenvalues;
    for (auto &v : eigenvalues) v *= eigenvalue_scale;

    auto modes = PostprocessModes(
        eigenvalues, summary.Shapes, float(1 / std::sqrt(rho_ratio)), material, config, current.Positions
    );
    modes.BakedScale = current.BakedScale;
    return modes;
}

modal::ModalResult modal::BuildModalResult(
    std::vector<double> eigenvalues, std::vector<std::vector<vec3>> shapes,
    const AcousticMaterialProperties &material, const SolverConfig &config,
    std::vector<vec3> positions, vec3 baked_scale, MassProperties mass_properties,
    SolveProfile profile, numeric::Matrix<float> basis, std::vector<uint32_t> sample_point_of
) {
    ModalEigenSummary summary{
        .Eigenvalues = std::move(eigenvalues),
        .Shapes = std::move(shapes),
        .SolvedMaterial = material,
    };
    auto modes = PostprocessModes(summary.Eigenvalues, summary.Shapes, 1, material, config, std::move(positions));
    modes.BakedScale = baked_scale;
    return {
        std::move(modes), std::move(mass_properties), profile, std::move(summary),
        std::move(basis), std::move(sample_point_of)
    };
}
