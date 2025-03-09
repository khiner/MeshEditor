#pragma once

#include <vector>

struct ModalModes {
    std::vector<float> Freqs; // Mode frequencies
    std::vector<float> T60s; // Mode T60 decay times
    std::vector<std::vector<float>> Gains; // Mode gains by [exitation position][mode]
};
