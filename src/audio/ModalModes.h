#pragma once

#include "numeric/vec3.h"

#include <vector>

struct ModalModes {
    std::vector<float> Freqs; // Mode frequencies
    std::vector<float> T60s; // Mode T60 decay times
    std::vector<std::vector<vec3>> Shapes; // Mass-normalized mode shape vectors by [excitation position][mode]
    std::vector<uint32_t> Vertices; // Mesh vertex indices corresponding to each excitation position in Shapes
    std::vector<vec3> Positions; // Node-local sample point positions, one per excitation position in Shapes
    float OriginalFundamentalFreq{Freqs.empty() ? 0 : Freqs.front()}; // Unscaled fundamental frequency, directly from FEM
    vec3 BakedScale{1.f}; // Node world-scale the modes were baked at. Later uniform scale changes retune relative to this.

    bool operator==(const ModalModes &) const = default;
};

// Per-instance
struct ModalGain {
    float Value{1.f};
};

// Per-instance synth tuning.
struct ModalTuning {
    float FundamentalFreq{0.f}; // Target frequency of the first mode, Hz. All modes shift proportionally. 0 keeps the baked tuning.
    float T60Scale{1.f}; // Multiplies every mode's T60.
};
