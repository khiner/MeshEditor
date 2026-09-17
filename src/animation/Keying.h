#pragma once

#include "animation/AnimationData.h"
#include "state/Entity.h"

namespace state {
struct Scene;
} // namespace state

namespace animation {
// A channel target on the entity holding its clips.
struct KeyTarget {
    state::Entity Entity;
    ChannelTarget Target;
};

// Seconds of a timeline frame in the viewport's frame rate.
float FrameSeconds(const state::Scene &, state::Entity viewport, int frame);

struct ChannelState {
    bool HasChannel{}, KeyAtFrame{}, Changed{};
};
ChannelState QueryChannel(const state::Scene &, state::Entity viewport, const KeyTarget &, float seconds);

// Keys the target's current value at `seconds`, creating its clip in the active animation and its channel when missing.
void InsertKey(state::Scene &, state::Entity viewport, const KeyTarget &, float seconds);
// Removes the key at `seconds`, and the channel, clip, and component that become empty.
bool DeleteKey(state::Scene &, state::Entity viewport, const KeyTarget &, float seconds);
// Deletes every key of the entity's active clip at `seconds`.
bool DeleteKeys(state::Scene &, state::Entity viewport, state::Entity, float seconds);

// Whether any animated field's value differs from its channel at `seconds`.
bool AnyChanged(const state::Scene &, state::Entity viewport, float seconds);
// Keys every animated field whose value differs from its channel at `seconds`.
void RecordChanged(state::Scene &, state::Entity viewport, float seconds);

// Writes the rest value back to each field animation `from` animates and animation `to` does not.
void RestoreRest(state::Scene &, uint32_t from, uint32_t to);
} // namespace animation
