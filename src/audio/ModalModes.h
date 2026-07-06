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
};

// Per-instance
struct ModalGain {
    float Value{1.f};
};
