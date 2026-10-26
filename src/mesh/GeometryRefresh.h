#pragma once

#include "gpu/RetessellateFacesPushConstants.h"
#include "mesh/PositionOperations.h"

struct MeshRetessellation {
    uint32_t StoreId;
    const BufferArena<uint32_t> *Work;
    ElementWork Faces;
    RetessellateFacesPushConstants Parameters;
};

// Canonical tessellation/tangent maintenance shared by procedural and editor publication.
// Each result is its input's changed triangles, work in chain.Scratch whose count the chain's next submit reports, or no work for an input without polygons to retessellate.
std::vector<ElementWork> EncodeRetessellation(state::Registry &, mtl::ComputeChain &, std::span<const MeshRetessellation>);
// Completes position operations, including sparse canonical tessellation, normals and bounds.
std::vector<PositionOperationChange> ExecuteGeometryPositions(state::Registry &, std::span<const PositionOperationTarget>, PositionEditOp, float factor, uint32_t repeat = 1u, const PositionOperationOptions & = {});
