#pragma once
#include "viewport/ViewportRenderGpu.h"

// The serial of the last submitted frame, which on the one queue completes after every earlier commit.
struct ViewportRenderResources {
    uint64_t InFlight{};
    bool RenderPending{false}; // A submitted frame has not been waited on.
};

// Commits the viewport's command recording the phase, whose cull saw every disocclusion so far.
// A phase other than Prepare takes the viewport's render request and records its scene as rendered.
void SubmitRecordedFrame(state::Registry &, state::Entity viewport, mtl::Command &&, RenderPhase);
// Records and submits the phase from the viewport's view uniforms.
void RecordAndSubmitFrame(state::Registry &, state::Entity viewport, RenderPhase = RenderPhase::Full);
