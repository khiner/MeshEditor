#pragma once

#include "gpu/Anisotropy.h"
#include "gpu/Clearcoat.h"
#include "gpu/DiffuseTransmission.h"
#include "gpu/Iridescence.h"
#include "gpu/MaterialAlphaMode.h"
#include "gpu/Sheen.h"
#include "gpu/Specular.h"
#include "gpu/TextureInfo.h"
#include "gpu/Transmission.h"
#include "gpu/Types.h"
#include "gpu/Volume.h"

struct PBRMaterial {
    vec4 BaseColorFactor DEFAULT(1);
    vec3 EmissiveFactor DEFAULT(0);
    float MetallicFactor DEFAULT(1);
    float RoughnessFactor DEFAULT(1);
    float NormalScale DEFAULT(1);
    float OcclusionStrength DEFAULT(1);
    MaterialAlphaMode AlphaMode DEFAULT(MaterialAlphaMode::Opaque);
    float AlphaCutoff DEFAULT(0.5);
    uint32_t DoubleSided DEFAULT();
    uint32_t Unlit DEFAULT();
    float Ior DEFAULT(1.5);
    float Dispersion DEFAULT(0);
    TextureInfo BaseColorTexture DEFAULT();
    TextureInfo MetallicRoughnessTexture DEFAULT();
    TextureInfo NormalTexture DEFAULT();
    TextureInfo OcclusionTexture DEFAULT();
    TextureInfo EmissiveTexture DEFAULT();
    Sheen Sheen DEFAULT();
    Specular Specular DEFAULT();
    Transmission Transmission DEFAULT();
    DiffuseTransmission DiffuseTransmission DEFAULT();
    Volume Volume DEFAULT();
    Clearcoat Clearcoat DEFAULT();
    Anisotropy Anisotropy DEFAULT();
    Iridescence Iridescence DEFAULT();
};
static_assert(sizeof(PBRMaterial) == 708, "PBRMaterial size");
