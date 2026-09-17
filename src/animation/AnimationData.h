#pragma once

#include "numeric/quat.h"
#include "state/Schema.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// How a channel's floats interpolate: Float lerps, Quaternion slerps, and Bool steps between 0 and 1.
enum class ValueKind : uint8_t {
    Float,
    Quaternion,
    Bool,
};

enum class AnimationInterpolation : uint8_t {
    Step,
    Linear,
    CubicSpline
};

// The field a channel animates: Count floats at byte Offset of a store, addressed like an Update.
// Component keys the store: a component on the clip's entity, the material buffer, or the entity's morph weights.
// Index selects the material in the material buffer. Values hold the field's native units and layout.
struct ChannelTarget {
    state::TypeKey Component;
    uint16_t Offset{0};
    uint16_t Index{0};
    uint16_t Count{0};
    ValueKind Kind{ValueKind::Float};

    bool operator==(const ChannelTarget &) const = default;
};

// A quaternion channel value is quat's own bytes.
static_assert(sizeof(quat) == 4 * sizeof(float) && std::is_trivially_copyable_v<quat>);
inline quat LoadQuat(const float *v) {
    quat q;
    std::memcpy(&q, v, sizeof q);
    return q;
}
inline void StoreQuat(float *v, quat q) { std::memcpy(v, &q, sizeof q); }

// Keys in seconds with Count values per key, or in tangent, value, and out tangent per key for cubic channels.
// Rest is the field's value from before the channel animated it, written back when the channel stops driving the field.
struct AnimationChannel {
    ChannelTarget Target;
    AnimationInterpolation Interp{AnimationInterpolation::Linear};
    std::vector<float> Times;
    std::vector<float> Values;
    std::vector<float> Rest;
};

// An entity's channels in one scene animation.
struct AnimationClip {
    uint32_t Animation;
    std::vector<AnimationChannel> Channels;
};

struct AnimationClips {
    std::vector<AnimationClip> Clips;

    // The clip in scene animation `animation`, or null.
    auto *Find(this auto &self, uint32_t animation) {
        const auto it = std::ranges::find(self.Clips, animation, &AnimationClip::Animation);
        return it != self.Clips.end() ? &*it : nullptr;
    }
};

// The scene's animations. Active selects the animation every clip evaluates and records into.
// Record keys each changed animated property when a user edit commits.
struct Animations {
    std::vector<std::string> Names;
    uint32_t Active{0};
    bool Record{false};
};
