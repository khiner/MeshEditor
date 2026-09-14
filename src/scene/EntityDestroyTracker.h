#pragma once

#include "state/Scene.h"

struct RenderInstance;

// Unmanaged reactive storage tracking entity destruction.
// Unlike managed dirty sets, this keeps destroyed entities until manually cleared, so deletions stay observable across a frame.
struct EntityDestroyTracker {
    state::DirtySet Storage;

    void Bind(state::Scene &r) {
        Storage.bind(r);
        Storage.on<RenderInstance>(state::On::Destroy);
    }
};
