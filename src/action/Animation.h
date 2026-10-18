#pragma once

#include "action/Core.h"
#include "animation/AnimationData.h"

namespace action::animation {
// Selected targets are bones in Pose mode and objects otherwise.
// An absent target inserts transform keys or deletes keys from every channel.
struct KeyScope {
    Target Target{OnActive{}};
    std::optional<ChannelTarget> Channel{};
};
struct InsertKey {
    KeyScope Keys;
};
struct DeleteKey {
    KeyScope Keys;
};
// Keys every animated field whose value differs from its channel at the current frame.
struct RecordChanged {};
struct AddAnimation {
    std::string Name;
};
struct RenameAnimation {
    uint32_t Index;
    std::string Name;
};
// Activate the animation at its start frame and set the timeline end to its last key.
// Reset fields animated only by the previous animation to rest.
struct SelectAnimation {
    uint32_t Index;
};

using Action = std::variant<InsertKey, DeleteKey, RecordChanged, AddAnimation, RenameAnimation, SelectAnimation>;

void Apply(state::Registry &, state::Entity viewport, const Action &);
} // namespace action::animation
