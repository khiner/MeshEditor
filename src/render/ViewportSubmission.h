#pragma once
#include "viewport/ViewportRenderGpu.h"
// Metal command buffers are single-use, and RecordedPhase tracks the last build.
struct ViewportRenderResources {
    MTL::CommandBuffer *InFlight{nullptr}; // The submitted frame, until it completes.
    RenderPhase RecordedPhase{RenderPhase::Full};
};

void SubmitRecordedFrame(state::Scene &, MTL::CommandBuffer *);
void RecordAndSubmitFrame(state::Scene &, state::Entity, SceneUpdate, RenderPhase = RenderPhase::Full);
