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
    float EmissiveStrength DEFAULT(1);
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
static_assert(sizeof(PBRMaterial) == 712, "PBRMaterial size");
#ifndef __METAL_VERSION__
#include "Field.h"
template<> inline constexpr FieldSpec Spec<PBRMaterial, "MetallicFactor">{.Min = 0, .Max = 1};
template<> inline constexpr FieldSpec Spec<PBRMaterial, "RoughnessFactor">{.Min = 0, .Max = 1};
template<> inline constexpr FieldSpec Spec<PBRMaterial, "NormalScale">{.Min = -2, .Max = 2};
template<> inline constexpr FieldSpec Spec<PBRMaterial, "OcclusionStrength">{.Min = 0, .Max = 1};
template<> inline constexpr FieldSpec Spec<PBRMaterial, "AlphaCutoff">{.Min = 0, .Max = 1};
template<> inline constexpr FieldSpec Spec<PBRMaterial, "Ior">{.Min = 1, .Max = 3};
template<> inline constexpr FieldSpec Spec<PBRMaterial, "Dispersion">{.Min = 0, .Max = 1};
#endif
