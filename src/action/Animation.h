#pragma once

#include "action/Core.h"
#include "animation/AnimationData.h"

#include <optional>
#include <string>

namespace action::animation {
// The keys an insert or delete addresses at the current frame.
// Selected targets are the selected bones in Pose mode and the selected objects otherwise.
// No target means translation, rotation, and scale for an insert and every channel for a delete.
struct KeyScope {
    Target Target{OnActive{}};
    std::optional<ChannelTarget> Channel{};
};
// Keys each target's current value.
struct InsertKey {
    KeyScope Keys;
};
struct DeleteKey {
    KeyScope Keys;
};
// Keys every animated field whose value differs from its channel at the current frame.
struct RecordChanged {};
// Appends a scene animation and makes it active.
struct AddAnimation {
    std::string Name;
};
struct RenameAnimation {
    uint32_t Index;
    std::string Name;
};
// Activate the animation, end the timeline at its last key, and jump to the start frame.
// Fields the previous animation animated and this one does not return to rest.
struct SelectAnimation {
    uint32_t Index;
};

using Action = std::variant<InsertKey, DeleteKey, RecordChanged, AddAnimation, RenameAnimation, SelectAnimation>;

void Apply(state::Scene &, state::Entity viewport, const Action &);
} // namespace action::animation
