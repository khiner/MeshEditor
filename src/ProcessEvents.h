#pragma once

#include <entt/entity/fwd.hpp>

enum class RenderRequest : uint8_t {
    None,
    Submit,
    ReRecordSilhouette, // Only silhouette batch + command buffer
    ReRecord, // Full draw list rebuild
};

// The strongest render request not yet handled by a record/submit.
struct PendingRenderRequest {
    RenderRequest Value{RenderRequest::None};
};

// Registers the reactive trackers ProcessComponentEvents drains and this system's scene setup/clear handlers.
void RegisterSceneComponentHandlers(entt::registry &);

// Drains Pending* markers, reactive trackers, and dirty sets accumulated since the last frame.
// Render work triggered by the drained changes accumulates in PendingRenderRequest.
// **This is the only place where direct by-ref component mutations (instead of `patch`) are allowed.**
void ProcessComponentEvents(entt::registry &, entt::entity viewport);
