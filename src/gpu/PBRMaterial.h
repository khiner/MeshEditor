#pragma once

#include "gpu/Anisotropy.h"
#include "gpu/Clearcoat.h"
#include "gpu/DiffuseTransmission.h"
#include "gpu/Iridescence.h"
#include "gpu/MaterialAlphaMode.h"
#include "gpu/Sheen.h"
#include "gpu/Specular.h"
#include "gpu/Transmission.h"
#include "gpu/Volume.h"

struct PBRMaterial {
    vec4 BaseColorFactor DEFAULT(1);
    TextureInfo BaseColorTexture DEFAULT();
    float MetallicFactor DEFAULT(1);
    float RoughnessFactor DEFAULT(1);
    TextureInfo MetallicRoughnessTexture DEFAULT();
    float NormalScale DEFAULT(1);
    TextureInfo NormalTexture DEFAULT();
    float OcclusionStrength DEFAULT(1);
    TextureInfo OcclusionTexture DEFAULT();
    vec3 EmissiveFactor DEFAULT(0);
    float EmissiveStrength DEFAULT(1);
    TextureInfo EmissiveTexture DEFAULT();
    MaterialAlphaMode AlphaMode DEFAULT(MaterialAlphaMode::Opaque);
    float AlphaCutoff DEFAULT(0.5);
    uint32_t DoubleSided DEFAULT();
    uint32_t Unlit DEFAULT();
    float Ior DEFAULT(1.5);
    float Dispersion DEFAULT(0);
    Sheen Sheen DEFAULT();
    Specular Specular DEFAULT();
    Transmission Transmission DEFAULT();
    DiffuseTransmission DiffuseTransmission DEFAULT();
    Volume Volume DEFAULT();
    Clearcoat Clearcoat DEFAULT();
    Anisotropy Anisotropy DEFAULT();
    Iridescence Iridescence DEFAULT();
};
#ifndef __METAL_VERSION__
template<> inline constexpr field::SpecTable<PBRMaterial> Specs<PBRMaterial>{
    {&PBRMaterial::BaseColorFactor, {.Widget = FieldWidget::Color, .Label = "Base color", .Help = "Multiplier on the base color texture. Alpha is coverage."}},
    {&PBRMaterial::BaseColorTexture, {.Help = "sRGB base color, multiplied by the base color factor"}},
    {&PBRMaterial::MetallicFactor, {.Min = 0, .Max = 1, .Label = "Metallic", .Help = "Metalness, multiplied by the texture's blue channel"}},
    {&PBRMaterial::RoughnessFactor, {.Min = 0, .Max = 1, .Label = "Roughness", .Help = "Roughness, multiplied by the texture's green channel"}},
    {&PBRMaterial::MetallicRoughnessTexture, {.Help = "Roughness in the green channel and metalness in the blue channel"}},
    {&PBRMaterial::NormalScale, {.Min = -2, .Max = 2, .Help = "Multiplier on the sampled normal's X and Y"}},
    {&PBRMaterial::NormalTexture, {.Help = "Tangent-space normal map"}},
    {&PBRMaterial::OcclusionStrength, {.Min = 0, .Max = 1, .Help = "Ambient occlusion amount"}},
    {&PBRMaterial::OcclusionTexture, {.Help = "Ambient occlusion in the red channel"}},
    {&PBRMaterial::EmissiveFactor, {.Widget = FieldWidget::Color, .Label = "Emissive", .Help = "Multiplier on the emissive texture"}},
    {&PBRMaterial::EmissiveStrength, {.Min = 0, .Speed = 0.01f, .Widget = FieldWidget::Drag, .Help = "Multiplier on the emitted light"}},
    {&PBRMaterial::EmissiveTexture, {.Help = "sRGB emitted color, multiplied by the emissive factor"}},
    {&PBRMaterial::AlphaMode, {.Help = "Opaque ignores alpha, Mask cuts out below the alpha cutoff, and Blend composites by alpha"}},
    {&PBRMaterial::AlphaCutoff, {.Min = 0, .Max = 1, .Help = "Alpha below which Mask mode discards the surface"}},
    {&PBRMaterial::DoubleSided, {.Widget = FieldWidget::Checkbox, .Help = "Render back faces, lit with flipped normals"}},
    {&PBRMaterial::Unlit, {.Widget = FieldWidget::Checkbox, .Help = "Shade with the base color alone, ignoring lighting"}},
    {&PBRMaterial::Ior, {.Min = 1, .Max = 3, .Label = "IOR", .Help = "Index of refraction"}},
    {&PBRMaterial::Dispersion, {.Min = 0, .Max = 1, .Help = "Spread of the index of refraction across wavelengths, as 20 over the Abbe number"}},
    {&PBRMaterial::Sheen, {.Help = "Back-scattering layer of cloth-like materials (KHR_materials_sheen)"}},
    {&PBRMaterial::Specular, {.Help = "Strength and color of dielectric specular reflection (KHR_materials_specular)"}},
    {&PBRMaterial::Transmission, {.Help = "Light passing through the surface, as in glass (KHR_materials_transmission)"}},
    {&PBRMaterial::DiffuseTransmission, {.Help = "Light scattering through thin surfaces, as in leaves and paper (KHR_materials_diffuse_transmission)"}},
    {&PBRMaterial::Volume, {.Help = "Thickness and absorption of the volume behind a transmissive surface (KHR_materials_volume)"}},
    {&PBRMaterial::Clearcoat, {.Help = "Clear reflective layer over the base material (KHR_materials_clearcoat)"}},
    {&PBRMaterial::Anisotropy, {.Help = "Directional stretching of specular highlights, as in brushed metal (KHR_materials_anisotropy)"}},
    {&PBRMaterial::Iridescence, {.Help = "Thin-film interference that shifts color with view angle (KHR_materials_iridescence)"}},
};
#endif
