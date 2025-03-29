#pragma once

#include <vector>

struct ModalModes {
    std::vector<float> Freqs; // Mode frequencies
    std::vector<float> T60s; // Mode T60 decay times
    std::vector<std::vector<float>> Gains; // Mode gains by [exitation position][mode]
    float OriginalFundamentalFreq{Freqs.empty() ? 0 : Freqs.front()}; // Unscaled fundamental frequency, directly from FEM
};
