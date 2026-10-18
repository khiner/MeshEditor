#pragma once

#include "selection/SelectionSchema.h"
#include "state/DirtySet.h"

// Selection change sets, bound by RegisterSceneComponentHandlers.
struct SelectionChanges {
    state::DirtySet Selected, ActiveInstance, BoneSelection, MeshActiveElement;
};
