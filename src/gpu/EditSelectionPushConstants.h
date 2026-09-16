#pragma once

#include "gpu/EditSelectionOperation.h"
#include "gpu/EditSelectionStorage.h"
#include "gpu/Element.h"
#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

struct EditSelectionPushConstants {
    EditSelectionStorage Selection DEFAULT();
    SlotOffset EdgeIndices DEFAULT();
    SlotOffset Corners DEFAULT();
    SlotOffset Connectivity DEFAULT();
    SlotOffset HalfedgeToEdge DEFAULT();
    SlotOffset EdgeHalfedges DEFAULT();
    SlotOffset Vertices DEFAULT();
    uint32_t VertexFanAdjacencyOffset DEFAULT(InvalidOffset);
    uint32_t VertexEdgeAdjacencyOffset DEFAULT(InvalidOffset);
    uint32_t AdjacencySlot DEFAULT(InvalidSlot);
    SlotOffset FaceSharpness DEFAULT();
    SlotOffset EdgeSharpness DEFAULT();
    SlotOffset SelectionBaseline DEFAULT();
    uint32_t VertexCount DEFAULT();
    uint32_t EdgeCount DEFAULT();
    uint32_t FaceCount DEFAULT();
    uint32_t HalfedgeCount DEFAULT();
    Element Element DEFAULT();
    uint32_t ConnectivityFaceStarts DEFAULT();
    EditSelectionOperation Operation DEFAULT();
    uint32_t PickIdSlot DEFAULT(InvalidSlot);
    SlotOffset SelectionList DEFAULT();
    uint32_t SelectionListCount DEFAULT();
    uint32_t PositionSumsSlot DEFAULT(InvalidSlot);
};
static_assert(sizeof(EditSelectionPushConstants) == 164, "EditSelectionPushConstants size");
