#pragma once

#include "numeric/vec3.h"

#include <vector>

struct ModalModes {
    std::vector<float> Freqs;
    std::vector<float> T60s;
    std::vector<std::vector<vec3>> Shapes; // Stores mass-normalized vectors by [excitation position][mode].
    std::vector<vec3> Positions; // Stores one node-local sample position per Shapes row.
    float OriginalFundamentalFreq{Freqs.empty() ? 0 : Freqs.front()}; // Stores the unscaled FEM fundamental frequency.
    vec3 BakedScale{1.f}; // Stores the node world scale used by the solve.

    bool operator==(const ModalModes &) const = default;
};
