#pragma once

#include <FastFEM/Surface2Modes.h>

#include "Field.h"
#include "numeric/vec3.h"

#include <vector>

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
}

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
template<> inline constexpr FieldSpec Spec<ModalGain, "Value">{.Min = 0, .Max = 2};

// Per-instance synth tuning.
struct ModalTuning {
    float FundamentalFreq{0.f}; // Target frequency of the first mode, Hz. All modes shift proportionally. 0 keeps the baked tuning.
    float T60Scale{1.f}; // Multiplies every mode's T60.
};
template<> inline constexpr FieldSpec Spec<ModalTuning, "FundamentalFreq">{.Min = 20, .Max = 16000, .Digits = 1};
template<> inline constexpr FieldSpec Spec<ModalTuning, "T60Scale">{.Min = 0.1, .Max = 10};
