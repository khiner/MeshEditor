#pragma once
#include "viewport/ViewportRenderGpu.h"
#include <Foundation/NSSharedPtr.hpp>
// Metal command buffers are single-use, and RecordedPhase tracks the last build.
struct ViewportRenderResources {
    ViewportRenderResources();
    ~ViewportRenderResources();
    NS::SharedPtr<MTL::CommandBuffer> InFlight;
    RenderPhase RecordedPhase{RenderPhase::Full};
};

void SubmitRecordedFrame(state::Scene &, MTL::CommandBuffer *);
void RecordAndSubmitFrame(state::Scene &, state::Entity, SceneUpdate, RenderPhase = RenderPhase::Full);
