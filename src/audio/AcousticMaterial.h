#pragma once

#include "AcousticMaterialProperties.h"
#include "Field.h"
#include "audio/AudioSchema.h"

struct AcousticMaterial {
    std::string Name;
    AcousticMaterialProperties Properties;

    bool operator==(const AcousticMaterial &) const = default;
};
// Ranges cover the acoustic material presets with headroom.
// Limit Poisson ratio below 0.5 to keep Lame's lambda finite.
// Set the beta floor above the logarithmic slider's zero epsilon.
template<> inline constexpr field::SpecTable<AcousticMaterialProperties> Specs<AcousticMaterialProperties>{
    {&AcousticMaterialProperties::Density, {.Min = 1, .Max = 25000, .Unit = FieldUnit::KilogramsPerCubicMeter, .Label = "Density", .Help = "Mass per unit volume"}},
    {&AcousticMaterialProperties::YoungModulus, {.Min = 1e5, .Max = 1e12, .Unit = FieldUnit::Pascals, .Scale = FieldScale::Logarithmic, .Label = "Young's modulus", .Format = "%.3g", .Help = "Stiffness in tension. Higher values raise every mode frequency."}},
    {&AcousticMaterialProperties::PoissonRatio, {.Min = 0, .Max = 0.49, .Label = "Poisson's ratio", .Help = "Ratio of lateral contraction to axial extension"}},
    {&AcousticMaterialProperties::Alpha, {.Min = 0, .Max = 200, .Unit = FieldUnit::PerSecond, .Label = "Rayleigh alpha", .Format = "%.3g", .Help = "Mass-proportional Rayleigh damping coefficient. Mostly damps low frequencies."}},
    {&AcousticMaterialProperties::Beta, {.Min = 1e-9, .Max = 1e-4, .Unit = FieldUnit::Seconds, .Scale = FieldScale::Logarithmic, .Label = "Rayleigh beta", .Format = "%.3g", .Help = "Stiffness-proportional Rayleigh damping coefficient. Mostly damps high frequencies."}},
};

// Material presets from KleinPAT, Table 4.
// https://graphics.stanford.edu/projects/kleinpat/assets/mfpat_opt.pdf

namespace materials::acoustic {
constexpr AcousticMaterial
    Ceramic{"Ceramic", {2700, 7.2E10, 0.19, 6, 1E-7}},
    Glass{"Glass", {2600, 6.2E10, 0.20, 1, 1E-7}},
    Wood{"Wood", {750, 1.1E10, 0.25, 60, 2E-6}},
    Plastic{"Plastic", {1070, 1.4E9, 0.35, 30, 1E-6}},
    Iron{"Iron", {8000, 2.1E11, 0.28, 5, 1E-7}},
    Polycarbonate{"Polycarbonate", {1190, 2.4E9, 0.37, 0.5, 4E-7}},
    Steel{"Steel", {7850, 2.0E11, 0.29, 5, 3E-8}};

constexpr std::array All{Ceramic, Glass, Wood, Plastic, Iron, Polycarbonate, Steel};

// The material with this name, or null.
inline const AcousticMaterial *Find(std::string_view name) {
    for (const auto &m : All) {
        if (m.Name == name) return &m;
    }
    return nullptr;
}
} // namespace materials::acoustic
