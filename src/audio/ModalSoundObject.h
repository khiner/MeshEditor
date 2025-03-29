#pragma once

#include "AcousticMaterial.h"
#include "Excitable.h"
#include "ModalModes.h"

#include <string_view>

static constexpr std::string_view ExciteIndexParamName{"Excite index"};
static constexpr std::string_view GateParamName{"Gate"};

// Assumes at least one mode is present.
struct ModalSoundObject {
    ModalModes Modes;
    Excitable Excitable;
};
