#pragma once

#include "animation/AnimationData.h"
#include "state/Entity.h"

#include <optional>
#include <span>

namespace state {
struct Scene;
} // namespace state

// Queries over clip data.
namespace animation {
// The viewport entity holding the scene animations, or Null before the viewport exists.
state::Entity AnimationsViewport(const state::Scene &);

// The entity's clip in the scene's active animation.
const AnimationClip *ActiveClip(const state::Scene &, state::Entity viewport, const AnimationClips &);

// Interpolates the channel at `seconds`, holding the first and last key outside the keyed range.
void EvaluateChannel(const AnimationChannel &, float seconds, std::span<float> out);

enum TransformComponentBit : uint8_t {
    TranslationBit = 1,
    RotationBit = 2,
    ScaleBit = 4,
};
// The transform components the active animation poses on `e`, as TransformComponentBits.
uint8_t PosedTransformComponents(const state::Scene &, state::Entity viewport, state::Entity e);

// The time of the last key in `animation`, or in every animation when none is given.
float LastKeySeconds(const state::Scene &, std::optional<uint32_t> animation = {});
} // namespace animation
