#pragma once

#include "state/DirtySet.h"

// Mesh change sets, bound by RegisterSceneComponentHandlers.
struct MeshChanges {
    state::DirtySet Geometry, Material;
    state::DirtySet TetMesh, NewBufferEntity;
};
