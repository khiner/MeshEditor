#pragma once

#include "AnimationData.h"

// Component on scene object entities with imported glTF node TRS animation data.
// Clips target this entity only; channels use BoneIndex=0 and are evaluated against RestLocal.
struct NodeTransformAnimation {
    std::vector<AnimationClip> Clips;
    uint32_t ActiveClipIndex{0};
    Transform RestLocal{};
};
