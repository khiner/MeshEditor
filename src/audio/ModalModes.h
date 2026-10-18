#pragma once

#include <FastFEM/Surface2Modes.h>

#include "Field.h"
#include "audio/AudioSchema.h"

struct ModalModes : fastfem::ModalModes {
    std::vector<uint32_t> Vertices; // Mesh vertex indices corresponding to each excitation position in Shapes
    // Triangles over the sample points, three indices per triangle, forming the surface mode shapes interpolate over.
    // Empty when there is none.
    std::vector<uint32_t> Indices;

    bool operator==(const ModalModes &) const = default;
};

namespace zpp::bits {
template<std::size_t>
struct members;
} // namespace zpp::bits

auto serialize(const ModalModes &) -> zpp::bits::members<8>;

constexpr auto serialize(auto &archive, ModalModes &modes) {
    return archive(modes.Freqs, modes.T60s, modes.Shapes, modes.Vertices, modes.Positions, modes.Indices, modes.OriginalFundamentalFreq, modes.BakedScale);
}

constexpr auto serialize(auto &archive, const ModalModes &modes) {
    return archive(modes.Freqs, modes.T60s, modes.Shapes, modes.Vertices, modes.Positions, modes.Indices, modes.OriginalFundamentalFreq, modes.BakedScale);
}

// Per-instance
struct ModalGain {
    float Value{1.f};
};
template<> inline constexpr field::SpecTable<ModalGain> Specs<ModalGain>{{&ModalGain::Value, {.Min = 0, .Max = 2, .Label = "Gain", .Help = "Output gain of the object's modes"}}};

// Per-instance synth tuning.
struct ModalTuning {
    std::optional<float> FundamentalFreq{}; // Target frequency of the first mode, Hz. All modes shift proportionally. Empty keeps the baked tuning.
    float T60Scale{1.f}; // Multiplies every mode's T60.
};
template<> inline constexpr field::SpecTable<ModalTuning> Specs<ModalTuning>{
    {&ModalTuning::FundamentalFreq, {.Min = 20, .Max = 16000, .Initial = 440, .Unit = FieldUnit::Hertz, .Scale = FieldScale::Logarithmic, .Label = "Fundamental", .Help = "Target lowest mode frequency, shifting every mode proportionally. Absent keeps the solved tuning."}},
    {&ModalTuning::T60Scale, {.Min = 0.1, .Max = 10, .Help = "Multiplier on every mode's T60 decay time"}},
};
