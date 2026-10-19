#include "editor/Engine.h"
#include "editor/AudioIntegration.h"
#include "metal/MetalContext.h"
#include "viewport/Viewport.h"

Engine::Engine(bool audio) : Audio(audio) {
    R.Resources.emplace<mtl::Context>();
    P = std::make_unique<project::Project>(R);
    InitEngine(R);
    P->TrackStores();
    if (Audio) InitAudioSystem(R);
    SetupDocument(R);
}

Engine::~Engine() {
    WaitForRender(R);
    if (Audio) DeinitAudioSystem(R);
    P.reset();
    DeinitViewport(R);
}
