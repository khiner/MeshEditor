#pragma once
#include "gpu/Types.h"

enum MeshAttributeBit : uint32_t {
    MeshAttributeBit_Normal = 1u << 0,
    MeshAttributeBit_Tangent = 1u << 1,
    MeshAttributeBit_Color0 = 1u << 2,
    MeshAttributeBit_TexCoord0 = 1u << 3,
    MeshAttributeBit_TexCoord1 = 1u << 4,
    MeshAttributeBit_TexCoord2 = 1u << 5,
    MeshAttributeBit_TexCoord3 = 1u << 6,
};
