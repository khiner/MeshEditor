#include "editor/Engine.h"
#include "editor/AudioIntegration.h"
#include "metal/MetalContext.h"
#include "viewport/Viewport.h"

Engine::Engine(bool audio) : Audio(audio) {
    R.ctx().emplace<mtl::Context>();
    P = std::make_unique<project::Project>(R);
    Viewport = InitEngine(R);
    P->TrackStores(Viewport);
    if (Audio) InitAudioSystem(R);
    SetupScene(R, Viewport);
}

Engine::~Engine() {
    WaitForRender(R);
    if (Audio) DeinitAudioSystem(R);
    P.reset();
    DeinitViewport(R, Viewport);
}
