#pragma once

#include <compare>

struct AcousticMaterialProperties {
    double Density, YoungModulus, PoissonRatio;
    double Alpha, Beta; // Rayleigh damping coefficients

    auto operator<=>(const AcousticMaterialProperties &) const = default;
};
