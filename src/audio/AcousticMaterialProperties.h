#pragma once

#include <compare>

// Isotropic material
struct AcousticMaterialProperties {
    double Density, YoungModulus, PoissonRatio;
    double Alpha, Beta; // Rayleigh damping coefficients

    // Lame's lambda coefficient
    double Lambda() const { return (PoissonRatio * YoungModulus) / ((1 + PoissonRatio) * (1 - 2 * PoissonRatio)); }
    // Lame's mu coefficient
    double Mu() const { return YoungModulus / (2 * (1 + PoissonRatio)); }

    auto operator<=>(const AcousticMaterialProperties &) const = default;
};
