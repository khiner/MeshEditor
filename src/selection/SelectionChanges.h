#pragma once

// Selection change sets, bound by RegisterSceneComponentHandlers.
struct SelectionChanges {
    state::DirtySet Selected, ActiveInstance, BoneSelection, MeshActiveElement;
};
