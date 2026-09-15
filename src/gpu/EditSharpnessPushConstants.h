#pragma once

#include "gpu/Types.h"
#include "gpu/SlotOffset.h"
#include "gpu/EditSharpnessOperation.h"

struct EditSharpnessPushConstants {
    SlotOffset VertexSelectionBits DEFAULT();
    SlotOffset EdgeSelectionBits DEFAULT();
    SlotOffset FaceSelectionBits DEFAULT();
    SlotOffset FaceSharpness DEFAULT();
    SlotOffset EdgeSharpness DEFAULT();
    SlotOffset Connectivity DEFAULT();
    SlotOffset EdgeHalfedges DEFAULT();
    SlotOffset EdgeIndices DEFAULT();
    SlotOffset FaceNormals DEFAULT();
    uint32_t VertexCount DEFAULT();
    uint32_t EdgeCount DEFAULT();
    uint32_t FaceCount DEFAULT();
    uint32_t HalfedgeCount DEFAULT();
    uint32_t ConnectivityFaceStarts DEFAULT();
    EditSharpnessOperation Operation DEFAULT();
    uint32_t Value DEFAULT();
    float CosAngle DEFAULT();
};
static_assert(sizeof(EditSharpnessPushConstants) == 104, "EditSharpnessPushConstants size");
