#pragma once

#include "gpu/ElementAttributeRef.h"
#include "gpu/ElementWork.h"
#include "gpu/GeometryEditMode.h"
#include "gpu/NormalDeriveEntry.h"
#include "gpu/SlotOffset.h"
#include "gpu/Transform.h"
#include "gpu/Types.h"

struct CommitPosedGeometryPushConstants {
    SlotOffset Vertices DEFAULT();
    uint32_t PositionSlot DEFAULT();
    uint32_t PositionNodesSlot DEFAULT();
    uint32_t SelectionSlot DEFAULT(InvalidSlot);
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
    uint32_t FaceTriangleStartSlot DEFAULT();
    ElementAttributeRef ElementMeshlets[3] DEFAULT();
    uint32_t Phase DEFAULT();
    uint32_t BudgetOffset DEFAULT();
    uint32_t ApplyTransform DEFAULT(1);
    GeometryEditMode Mode DEFAULT();
};
static_assert(sizeof(CommitPosedGeometryPushConstants) == 420, "CommitPosedGeometryPushConstants size");
