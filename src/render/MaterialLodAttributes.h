#pragma once

#include "gpu/DebugChannel.h"
#include "gpu/MeshAttributeBit.h"
#include "gpu/PBRMaterial.h"

#include <algorithm>

// Preserve declared material inputs even when a viewport disables an extension.
inline uint32_t MaterialLodAttributes(const PBRMaterial &material, DebugChannel debug, bool authored_tangents) {
    uint32_t attributes = MeshAttributeBit_Normal | MeshAttributeBit_Color0;
    const auto uv_bit = [](uint32_t set) { return uint32_t(MeshAttributeBit_TexCoord0) << std::min(set, 3u); };
    const TextureInfo *textures[]{
        &material.BaseColorTexture,
        &material.MetallicRoughnessTexture,
        &material.NormalTexture,
        &material.OcclusionTexture,
        &material.EmissiveTexture,
        &material.Sheen.ColorTexture,
        &material.Sheen.RoughnessTexture,
        &material.Specular.Texture,
        &material.Specular.ColorTexture,
        &material.Transmission.Texture,
        &material.DiffuseTransmission.Texture,
        &material.DiffuseTransmission.ColorTexture,
        &material.Volume.ThicknessTexture,
        &material.Clearcoat.Texture,
        &material.Clearcoat.RoughnessTexture,
        &material.Clearcoat.NormalTexture,
        &material.Anisotropy.Texture,
        &material.Iridescence.Texture,
        &material.Iridescence.ThicknessTexture,
    };
    for (const auto *texture : textures)
        if (texture->Slot != InvalidSlot) attributes |= uv_bit(texture->TexCoord);
    const bool basis = material.NormalTexture.Slot != InvalidSlot || material.Clearcoat.NormalTexture.Slot != InvalidSlot ||
        material.Anisotropy.Strength > 0.f || debug == DebugChannel::Tangent || debug == DebugChannel::Bitangent;
    if (basis) {
        // Zero authored tangents derive their frame from the base normal UV set.
        attributes |= uv_bit(material.NormalTexture.TexCoord);
        if (authored_tangents) attributes |= MeshAttributeBit_Tangent;
    }
    if (debug == DebugChannel::TangentW) attributes |= MeshAttributeBit_Tangent;
    if (debug == DebugChannel::UvCoords0) attributes |= MeshAttributeBit_TexCoord0;
    if (debug == DebugChannel::UvCoords1) attributes |= MeshAttributeBit_TexCoord1;
    return attributes;
}
