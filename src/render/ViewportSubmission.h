#pragma once
#include "render/RenderSchema.h"
#include "viewport/ViewportRenderGpu.h"
#include <Foundation/NSSharedPtr.hpp>
// Metal command buffers are single-use, and RecordedPhase tracks the last build.
struct ViewportRenderResources {
    ViewportRenderResources();
    ~ViewportRenderResources();
    NS::SharedPtr<MTL::CommandBuffer> InFlight;
    RenderPhase RecordedPhase{RenderPhase::Full};
};

void SubmitRecordedFrame(state::Registry &, MTL::CommandBuffer *);
void RecordAndSubmitFrame(state::Registry &, state::Entity, RenderPhase = RenderPhase::Full);
