#pragma once
#include "gpu/ConnectivityRef.h"
#include "gpu/ElementAttributeRef.h"
#include "gpu/ElementWork.h"

struct FaceAttributeEditPushConstants {
    ConnectivityRef Connectivity;
    ElementWork Faces;
    ElementAttributeRef Attribute, Tangents;
    uint32_t FaceCount, Count, Operation;
};
