#pragma once

#include "mesh/MeshSchema.h"
#include "state/DirtySet.h"

// Mesh change sets, bound by RegisterSceneComponentHandlers.
struct MeshChanges {
    state::DirtySet Geometry, Material;
    state::DirtySet PrimitiveShape; // A primitive mesh's shape fields changed.
    state::DirtySet TetMesh, NewBufferEntity;
};
