#pragma once

#include "state/Slot.h"

struct AnimationChanges;
struct AnimationClips;
struct Animations;
struct MorphWeightRange;

using AnimationSchema = state::Schema<state::AnimationBase, state::AnimationCount, AnimationChanges, AnimationClips, Animations, MorphWeightRange>;
template<typename T>
    requires AnimationSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = AnimationSchema::Of<T>();
