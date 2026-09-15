#pragma once

#include "gpu/Types.h"

enum class EditSelectionOperation : uint32_t {
    Derive = 0,
    Clear = 1,
    Fill = 2,
    FillList = 3,
    CaptureBaseline = 4,
    RestoreBaseline = 5,
    PickReplace = 6,
    PickToggle = 7,
    ClearActive = 8,
};
