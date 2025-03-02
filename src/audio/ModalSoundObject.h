#pragma once

#include "AcousticMaterial.h"
#include "Excitable.h"

#include <optional>
#include <string_view>

static constexpr std::string_view ExciteIndexParamName{"Excite index"};
static constexpr std::string_view GateParamName{"Gate"};

struct ModalSoundObject {
    std::vector<float> ModeFreqs; // Mode frequencies
    std::vector<float> ModeT60s; // Mode T60 decay times
    std::vector<std::vector<float>> ModeGains; // Mode gains by [exitation position][mode]
    ::Excitable Excitable;
    float FundamentalFreq{!ModeFreqs.empty() ? ModeFreqs.front() : 440.f}; // Override to scale mode frequencies
};
