#pragma once

#include "gpu/Types.h"
#include "gpu/SlotOffset.h"
#include "gpu/ElementWork.h"
#include "gpu/NormalDeriveEntry.h"
#include "gpu/Transform.h"
#include "gpu/GeometryEditMode.h"

struct CommitPosedGeometryPushConstants {
    SlotOffset Vertices DEFAULT();
    SlotOffset Output DEFAULT();
    SlotOffset Selection DEFAULT();
    ElementWork Candidates DEFAULT();
    ElementWork ChangedVertices DEFAULT();
    ElementWork Faces DEFAULT();
    ElementWork Normals DEFAULT();
    ElementWork Meshlets DEFAULT();
    ElementWork BoundsTiles DEFAULT();
    NormalDeriveEntry Entry DEFAULT();
    Transform Primary DEFAULT();
    Transform Delta DEFAULT();
    vec3 Pivot DEFAULT();
    uint32_t AdjacencySlot DEFAULT();
    uint32_t FaceFirstTriangleSlot DEFAULT();
    uint32_t CornerClassSlot DEFAULT();
    uint32_t CornerClassOffset DEFAULT();
    uint32_t VertexEdgeAdjacencyOffset DEFAULT(InvalidOffset);
    uint32_t Topology DEFAULT();
    SlotOffset TriangleMeshlets DEFAULT();
    uint32_t Phase DEFAULT();
    uint32_t ApplyTransform DEFAULT(1);
    GeometryEditMode Mode DEFAULT();
};
static_assert(sizeof(CommitPosedGeometryPushConstants) == 292, "CommitPosedGeometryPushConstants size");
