#pragma once

#include "state/Entity.h"

// Whether a viewport frame waits for its record and submit.
struct PendingRenderRequest {
    bool Value{false};
};

// Queue a viewport frame, which invalidates occlusion unless its change cannot disocclude anything, as a selection change cannot.
void RequestRender(state::Scene &, bool disoccludes = true);

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
// Process component changes and request a frame in PendingRenderRequest.
void ProcessComponentEvents(state::Scene &, state::Entity viewport, EventPass = EventPass::Frame);
