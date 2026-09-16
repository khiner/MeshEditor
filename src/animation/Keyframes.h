#pragma once

#include "state/Entity.h"

#include <vector>

// Timeline frames holding a key on any channel of the selected objects' active clips.
// Clips loop, so each key also repeats every clip duration through the timeline end frame.
// Sorted ascending without duplicates.
std::vector<float> CollectKeyframes(const state::Scene &, state::Entity viewport);
