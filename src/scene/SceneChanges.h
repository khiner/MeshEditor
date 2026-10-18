#pragma once

#include "scene/SceneSchema.h"
#include "state/DirtySet.h"

// Scene graph change sets, bound by RegisterSceneComponentHandlers.
struct SceneChanges {
    state::DirtySet WorldTransform; // A world transform or a bone's display scale changed.
    state::DirtySet Parent; // A node's parent changed.
    state::DirtySet Hierarchy; // A scene graph link changed.
    state::DirtySet Names; // A name was created or destroyed.
    state::DirtySet ScaleLocked, TransformDirty;
};
