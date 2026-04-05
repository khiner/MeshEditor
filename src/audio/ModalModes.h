#pragma once

#include <cstdint>
#include <vector>

struct ModalModes {
    std::vector<float> Freqs; // Mode frequencies
    std::vector<float> T60s; // Mode T60 decay times
    std::vector<std::vector<float>> Gains; // Mode gains by [exitation position][mode]
    std::vector<uint32_t> Vertices; // Mesh vertex indices corresponding to each excitation position in Gains
    float OriginalFundamentalFreq{Freqs.empty() ? 0 : Freqs.front()}; // Unscaled fundamental frequency, directly from FEM
};
