#pragma once

#include <entt/entity/fwd.hpp>

enum class RenderRequest : uint8_t {
    None,
    Submit,
    ReRecordSilhouette, // Only silhouette batch + command buffer
    ReRecord, // Full draw list rebuild
};

// Drains Pending* markers, reactive trackers, and dirty sets accumulated since the last frame.
// Returns the strongest render request triggered by this frame's changes.
RenderRequest ProcessComponentEvents(entt::registry &, entt::entity viewport);
