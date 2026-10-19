#pragma once
#include "render/RenderSchema.h"
#include "viewport/ViewportRenderGpu.h"
#include <Foundation/NSSharedPtr.hpp>

struct Viewport;

// The last submitted command buffer, which on the one queue completes after every earlier one.
struct ViewportRenderResources {
    ViewportRenderResources();
    ~ViewportRenderResources();
    NS::SharedPtr<MTL::CommandBuffer> InFlight;
    bool RenderPending{false}; // A submitted frame has not been waited on.
};

// Commits the viewport's command buffer recording the phase, whose cull saw every disocclusion so far.
// A phase other than Prepare takes the viewport's render request and records its state as rendered.
void SubmitRecordedFrame(state::Registry &, Viewport &, MTL::CommandBuffer *, RenderPhase);
// Records and submits the phase from the viewport's view uniforms.
void RecordAndSubmitFrame(state::Registry &, Viewport &, RenderPhase = RenderPhase::Full);
