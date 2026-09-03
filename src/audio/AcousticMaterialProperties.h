#pragma once

#include <compare>

struct AcousticMaterialProperties {
    double Density, YoungModulus, PoissonRatio;
    double Alpha, Beta; // Rayleigh damping coefficients

    double Lambda() const { return (PoissonRatio * YoungModulus) / ((1 + PoissonRatio) * (1 - 2 * PoissonRatio)); }
    double Mu() const { return YoungModulus / (2 * (1 + PoissonRatio)); }

    auto operator<=>(const AcousticMaterialProperties &) const = default;
};
