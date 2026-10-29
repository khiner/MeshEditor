#pragma once

#include "armature/ArmatureSchema.h"
#include "state/DirtySet.h"

// Bone change sets, bound by RegisterSceneComponentHandlers.
struct ArmatureChanges {
    state::DirtySet BonePose; // A bone's pose delta changed.
    state::DirtySet BoneConstraints; // A bone's constraint stack changed.
    state::DirtySet SkinBinding; // An object's armature or skin binding changed.
};
