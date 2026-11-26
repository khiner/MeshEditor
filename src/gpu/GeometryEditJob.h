#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"
#include "gpu/GeometryEditMode.h"
#include "gpu/MeshElementSet.h"

// One mesh's job in a geometry edit pass, whose phase is the pass parameter.
// Candidates holds the edited vertex handles, Faces and Normals the faces and vertices whose normals change, and MeshletAddresses the clusters they touch.
struct GeometryEditJob {
    ElementWork Candidates DEFAULT();
    ElementWork ChangedVertices DEFAULT();
    ElementWork Faces DEFAULT();
    ElementWork Normals DEFAULT();
    ElementWork MeshletAddresses DEFAULT();
    ConnectivityRef Connectivity DEFAULT();
    uint32_t FaceTriangleStartSlot DEFAULT();
    SlotOffset ElementMeshlets[3] DEFAULT(); // The record's element owner ranges, holding handles of its meshlet set
    SetArenaSlots MeshletSets DEFAULT();
    uint32_t MeshletSet DEFAULT(InvalidOffset);
    GeometryEditMode Mode DEFAULT();
};
