#pragma once

#include "gpu/Transform.h"

#include <span>
#include <string>
#include <vector>

using BoneId = uint32_t; // Stable identifier - never reused
inline constexpr BoneId InvalidBoneId{0};

enum class AnimationPath : uint8_t {
    Translation,
    Rotation,
    Scale,
    Weights
};

enum class AnimationInterpolation : uint8_t {
    Step,
    Linear,
    CubicSpline
};

// Generic animation channel.
// For bone channels, TargetBoneId stores the stable identity; BoneIndex is a resolved cache.
// For non-bone channels (e.g. node animation), TargetBoneId is InvalidBoneId and BoneIndex is used directly.
struct AnimationChannel {
    uint32_t BoneIndex;
    BoneId TargetBoneId{InvalidBoneId};
    AnimationPath Target;
    AnimationInterpolation Interp;
    std::vector<float> TimesSeconds;
    std::vector<float> Values;
};

struct AnimationClip {
    std::string Name;
    float DurationSeconds;
    std::vector<AnimationChannel> Channels;
};

// Morph weight animation channel — dedicated type (no BoneIndex/Target overhead).
struct MorphWeightChannel {
    AnimationInterpolation Interp{AnimationInterpolation::Linear};
    std::vector<float> TimesSeconds;
    std::vector<float> Values; // Packed: target_count floats per keyframe
};

struct MorphWeightClip {
    std::string Name;
    float DurationSeconds;
    std::vector<MorphWeightChannel> Channels;
};

// Interpolate animation channels at `time`, writing into pre-initialized rest-pose `local_transforms`.
void EvaluateAnimation(const AnimationClip &, float time, std::span<Transform> local_transforms);

// Interpolate morph weight animation channels at `time`, writing into `weights`.
void EvaluateMorphWeights(const MorphWeightClip &, float time, std::span<float> weights);
