#pragma once

#include "gpu/ElementWork.h"
#include "state/Entity.h"
#include <cstdint>
#include <span>

template<typename T> struct BufferArena;
struct MeshBuffers;
namespace state { struct Scene; }

struct MeshletBoundsRefitJob {
    MeshBuffers *Owner;
    const BufferArena<uint32_t> *Storage;
    ElementWork Meshlets;
};

// Refit only the finest meshlets affected by canonical vertex-position writes.
void RefitCanonicalMeshletBounds(state::Scene &,std::span<const MeshletBoundsRefitJob>);
// Refit the moved finest clusters' traversal leaves and mark their coarse ancestors stale.
void StageDirtyPositionMeshlets(state::Scene &,state::Entity);
