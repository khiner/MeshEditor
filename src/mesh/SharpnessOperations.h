#pragma once
#include "gpu/EditSharpnessOperation.h"
#include "mesh/GeometrySelection.h"
#include "mesh/MeshClosure.h"
#include <span>
#include <vector>

struct SharpnessOperationTarget {
    uint32_t StoreId;
    GeometrySelection Selection;
};
struct SharpnessOperationChange {
    uint32_t TargetIndex;
    FaceTriangles Triangles;
};
// Selected operations use explicit masks; all-face operations affect the whole mesh.
// Completes canonical sharpness, corner classification and normals. Triangle work
// remains in chain.Scratch for a caller's render publication.
std::vector<SharpnessOperationChange> ExecuteGeometrySharpness(state::Scene &, mtl::ComputeChain &, std::span<const SharpnessOperationTarget>, EditSharpnessOperation, bool value = false, float angle = 0.f);
