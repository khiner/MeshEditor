#pragma once

#include "state/Entity.h"
#include "viewport/ViewportSchema.h"

// Request a frame from every viewport, and invalidate each one's occlusion when the change can reveal hidden geometry.
void RequestRender(state::Registry &, bool disoccludes = true);

void RegisterSceneComponentHandlers(state::Registry &);

// Restore reconciles Derived state while preserving Persistent state.
// Sample evaluates a shutter time; Render restores the displayed pose, including live drag offsets.
enum class EventPass { Frame,
                       Settle,
                       Restore,
                       Sample,
                       Render };
void ProcessComponentEvents(state::Registry &, EventPass = EventPass::Frame);
