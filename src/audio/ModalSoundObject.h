#pragma once

#include "AcousticMaterial.h"
#include "Excitable.h"
#include "ModalModes.h"

#include <string_view>

static constexpr std::string_view ExciteIndexParamName{"Excite index"};
static constexpr std::string_view GateParamName{"Gate"};

struct ModalSoundObject {
    ModalModes Modes;
    Excitable Excitable;
    float FundamentalFreq{!Modes.Freqs.empty() ? Modes.Freqs.front() : 440.f}; // Override to scale mode frequencies
};
