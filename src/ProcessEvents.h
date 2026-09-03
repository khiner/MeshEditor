#pragma once

#include <entt/entity/fwd.hpp>

enum class RenderRequest : uint8_t {
    None,
    Reuse,
    Silhouette,
    Rebuild,
};

// The strongest render request not yet handled by a record/submit.
struct PendingRenderRequest {
    RenderRequest Value{RenderRequest::None};
};

// Register the reactive trackers and scene lifecycle handlers used by ProcessComponentEvents.
void RegisterSceneComponentHandlers(entt::registry &);

// Drain pending component changes and accumulate their render work in PendingRenderRequest.
// Direct component mutations outside action Apply handlers are restricted to this function.
void ProcessComponentEvents(entt::registry &, entt::entity viewport);
