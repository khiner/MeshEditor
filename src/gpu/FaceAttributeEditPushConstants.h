#pragma once
#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"

struct FaceAttributeEditPushConstants {
    ConnectivityRef Connectivity;
    ElementWork Faces;
    SlotOffset Attribute, Tangents;
    uint32_t FaceCount, Count, Operation;
};
