#pragma once

// Scene graph change sets, bound by RegisterSceneComponentHandlers.
struct SceneChanges {
    state::DirtySet WorldTransform; // A world transform or a bone's display scale changed.
    state::DirtySet Parent; // A node's parent changed.
    state::DirtySet Hierarchy; // A scene graph link changed.
    state::DirtySet Membership; // A root's scenes changed.
    state::DirtySet Names; // A name was created or destroyed.
    state::DirtySet Destroyed; // An object or bone was destroyed, kept until the selection aggregates drop it.
    state::DirtySet ScaleLocked, TransformDirty;
};
