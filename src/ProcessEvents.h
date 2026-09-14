#pragma once

#include "state/Entity.h"

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
void RegisterSceneComponentHandlers(state::Scene &);

// Restore reconciles Derived state without mutating Persistent state.
// Sample evaluates a shutter time.
// Render restores the displayed pose after sampling, including live drag offsets.
enum class EventPass { Frame,
                       Settle,
                       Restore,
                       Sample,
                       Render };
// Process component changes and accumulate render work in PendingRenderRequest.
void ProcessComponentEvents(state::Scene &, state::Entity viewport, EventPass = EventPass::Frame);
