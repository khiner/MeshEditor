#pragma once

#include "animation/AnimationSchema.h"
#include "state/DirtySet.h"

// Animation change sets, bound by RegisterSceneComponentHandlers.
struct AnimationChanges {
    state::DirtySet Edited;
    state::DirtySet KeyframeSources; // The timeline rate or a mesh's displayed material slot changed.
};
