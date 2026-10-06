#pragma once

#include "gpu/RetessellateFacesPushConstants.h"
#include "mesh/PositionOperations.h"

struct MeshRetessellation {
    uint32_t StoreId;
    const BufferArena<uint32_t> *Work;
    ElementWork Faces;
    RetessellateFacesPushConstants Parameters;
    bool Preview{};
};

// Canonical tessellation/tangent maintenance shared by procedural and editor publication.
// Result work lives in chain.Scratch; each result corresponds to its input.
std::vector<FaceTriangles> RetessellateMeshes(state::Scene &, mtl::ComputeChain &, std::span<const MeshRetessellation>);
// Completes position operations, including sparse canonical tessellation, normals and bounds.
std::vector<PositionOperationChange> ExecuteGeometryPositions(state::Scene &, std::span<const PositionOperationTarget>, PositionEditOp, float factor, uint32_t repeat = 1u, const PositionOperationOptions & = {});
