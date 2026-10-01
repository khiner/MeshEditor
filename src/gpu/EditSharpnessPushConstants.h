#pragma once
#include "gpu/ConnectivityRef.h"

#include "gpu/EditSharpnessOperation.h"
#include "gpu/Element.h"
#include "gpu/ElementWork.h"
#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

struct EditSharpnessPushConstants {
    uint32_t VertexSelectionSlot DEFAULT(InvalidSlot);
    uint32_t CornersSlot DEFAULT(InvalidSlot);
    uint32_t FaceSharpnessSlot DEFAULT(InvalidSlot);
    uint32_t EdgeSharpnessSlot DEFAULT(InvalidSlot);
    uint32_t FaceNormalsSlot DEFAULT(InvalidSlot);
    ConnectivityRef Connectivity DEFAULT();
    SlotOffset Selected DEFAULT();
    ElementWork FaceWork DEFAULT();
    ElementWork EdgeWork DEFAULT();
    uint32_t SelectedCount DEFAULT();
    uint32_t EdgeCount DEFAULT();
    uint32_t FaceCount DEFAULT();
    EditSharpnessOperation Operation DEFAULT();
    uint32_t Value DEFAULT();
    float CosAngle DEFAULT();
};
static_assert(sizeof(EditSharpnessPushConstants) == 144, "EditSharpnessPushConstants size");
