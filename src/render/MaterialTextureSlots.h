#pragma once

#include "gltf/SourceAssets.h"
#include "gpu/PBRMaterial.h"
#include "render/Textures.h"

#include <array>
#include <string_view>

// Top-level material texture slots (PBRMaterial accessor, color space, glTF label, glTF pointer path), ordered to match MaterialTextureSlot.
struct MaterialTextureSlotInfo {
    ::TextureInfo &(*Get)(PBRMaterial &);
    TextureColorSpace ColorSpace;
    std::string_view Label;
    std::string_view Pointer;
};
constexpr std::array<MaterialTextureSlotInfo, MTS_Count> MaterialTextureSlots{{
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.BaseColorTexture; }, TextureColorSpace::Srgb, "baseColor", "pbrMetallicRoughness/baseColorTexture"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.MetallicRoughnessTexture; }, TextureColorSpace::Linear, "metallicRoughness", "pbrMetallicRoughness/metallicRoughnessTexture"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.NormalTexture; }, TextureColorSpace::Linear, "normal", "normalTexture"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.OcclusionTexture; }, TextureColorSpace::Linear, "occlusion", "occlusionTexture"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.EmissiveTexture; }, TextureColorSpace::Srgb, "emissive", "emissiveTexture"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Specular.Texture; }, TextureColorSpace::Linear, "specular", "extensions/KHR_materials_specular/specularTexture"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Specular.ColorTexture; }, TextureColorSpace::Srgb, "specularColor", "extensions/KHR_materials_specular/specularColorTexture"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Sheen.ColorTexture; }, TextureColorSpace::Srgb, "sheenColor", "extensions/KHR_materials_sheen/sheenColorTexture"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Sheen.RoughnessTexture; }, TextureColorSpace::Linear, "sheenRoughness", "extensions/KHR_materials_sheen/sheenRoughnessTexture"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Transmission.Texture; }, TextureColorSpace::Linear, "transmission", "extensions/KHR_materials_transmission/transmissionTexture"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.DiffuseTransmission.Texture; }, TextureColorSpace::Linear, "diffuseTransmission", "extensions/KHR_materials_diffuse_transmission/diffuseTransmissionTexture"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.DiffuseTransmission.ColorTexture; }, TextureColorSpace::Srgb, "diffuseTransmissionColor", "extensions/KHR_materials_diffuse_transmission/diffuseTransmissionColorTexture"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Volume.ThicknessTexture; }, TextureColorSpace::Linear, "thickness", "extensions/KHR_materials_volume/thicknessTexture"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Clearcoat.Texture; }, TextureColorSpace::Linear, "clearcoat", "extensions/KHR_materials_clearcoat/clearcoatTexture"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Clearcoat.RoughnessTexture; }, TextureColorSpace::Linear, "clearcoatRoughness", "extensions/KHR_materials_clearcoat/clearcoatRoughnessTexture"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Clearcoat.NormalTexture; }, TextureColorSpace::Linear, "clearcoatNormal", "extensions/KHR_materials_clearcoat/clearcoatNormalTexture"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Anisotropy.Texture; }, TextureColorSpace::Linear, "anisotropy", "extensions/KHR_materials_anisotropy/anisotropyTexture"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Iridescence.Texture; }, TextureColorSpace::Linear, "iridescence", "extensions/KHR_materials_iridescence/iridescenceTexture"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Iridescence.ThicknessTexture; }, TextureColorSpace::Linear, "iridescenceThickness", "extensions/KHR_materials_iridescence/iridescenceThicknessTexture"},
}};
