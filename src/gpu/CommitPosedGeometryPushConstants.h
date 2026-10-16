#pragma once

#include "gpu/ElementWork.h"
#include "gpu/GeometryEditMode.h"
#include "gpu/MeshElementSet.h"
#include "gpu/NormalDeriveEntry.h"
#include "gpu/SlotOffset.h"
#include "gpu/Transform.h"
#include "gpu/Types.h"

// Work names the entry's vertex and face handles, MeshletAddresses its clusters, and BoundsTiles its vertex blocks in the arena.
struct CommitPosedGeometryPushConstants {
    uint32_t VerticesSlot DEFAULT(InvalidSlot);
    uint32_t PositionSlot DEFAULT();
    uint32_t PositionNodesSlot DEFAULT();
    uint32_t SelectionSlot DEFAULT(InvalidSlot);
    ElementWork Candidates DEFAULT();
    ElementWork ChangedVertices DEFAULT();
    ElementWork Faces DEFAULT();
    ElementWork Normals DEFAULT();
    ElementWork MeshletAddresses DEFAULT();
    ElementWork BoundsTiles DEFAULT();
    NormalDeriveEntry Entry DEFAULT();
    Transform Primary DEFAULT();
    Transform Delta DEFAULT();
    vec3 Pivot DEFAULT();
    uint32_t FaceTriangleStartSlot DEFAULT();
    SlotOffset ElementMeshlets[3] DEFAULT(); // The record's element owner ranges, holding handles of its meshlet set
    SetArenaSlots MeshletSets DEFAULT();
    uint32_t MeshletSet DEFAULT(InvalidOffset);
    uint32_t Phase DEFAULT();
    uint32_t BudgetOffset DEFAULT();
    uint32_t ApplyTransform DEFAULT(1);
    GeometryEditMode Mode DEFAULT();
};
