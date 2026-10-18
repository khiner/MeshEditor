#pragma once

#include "audio/AudioSchema.h"
#include "state/DirtySet.h"

// Sound vertex change sets, bound by RegisterSceneComponentHandlers.
struct SoundVertexChanges {
    state::DirtySet Vertices; // An object's sound vertices were created or destroyed.
    state::DirtySet Updated, VertexForce;
};
