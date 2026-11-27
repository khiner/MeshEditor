#pragma once

// Selection change sets, bound by RegisterSceneComponentHandlers.
struct SelectionChanges {
    state::DirtySet Selected, ActiveObject, BoneSelection, MeshActiveElement;
    state::DirtySet Editing; // A node's selection editing began or ended.
};
