#pragma once

#include "SourceAssets.h"
#include "animation/AnimationData.h"
#include "gpu/PBRMaterial.h"
#include "physics/PhysicsTypes.h"
#include "render/Textures.h"
#include <cstring>
#include <fastgltf/types.hpp>
#include <map>
#include <numbers>

namespace gltf::detail {
using ExtrasMap = std::map<uint64_t, std::string>;

inline uint64_t ExtrasKey(uint32_t category, size_t idx) { return (uint64_t(category) << 32) | uint64_t(idx); }

// MimeType, AlphaMode, and CombineMode share fastgltf's ordinals.
static_assert(uint8_t(fastgltf::MimeType::None) == uint8_t(MimeType::None) && uint8_t(fastgltf::MimeType::JPEG) == uint8_t(MimeType::JPEG) && uint8_t(fastgltf::MimeType::PNG) == uint8_t(MimeType::PNG) && uint8_t(fastgltf::MimeType::KTX2) == uint8_t(MimeType::KTX2) && uint8_t(fastgltf::MimeType::DDS) == uint8_t(MimeType::DDS) && uint8_t(fastgltf::MimeType::GltfBuffer) == uint8_t(MimeType::GltfBuffer) && uint8_t(fastgltf::MimeType::OctetStream) == uint8_t(MimeType::OctetStream) && uint8_t(fastgltf::MimeType::WEBP) == uint8_t(MimeType::WEBP));
inline MimeType ToMimeType(fastgltf::MimeType m) { return MimeType(uint8_t(m)); }
inline fastgltf::MimeType FromMimeType(MimeType m) { return fastgltf::MimeType(uint8_t(m)); }

static_assert(uint8_t(fastgltf::AlphaMode::Opaque) == uint8_t(MaterialAlphaMode::Opaque) && uint8_t(fastgltf::AlphaMode::Mask) == uint8_t(MaterialAlphaMode::Mask) && uint8_t(fastgltf::AlphaMode::Blend) == uint8_t(MaterialAlphaMode::Blend));
inline MaterialAlphaMode ToAlphaMode(fastgltf::AlphaMode m) { return MaterialAlphaMode(uint8_t(m)); }
inline fastgltf::AlphaMode FromAlphaMode(MaterialAlphaMode m) { return fastgltf::AlphaMode(uint8_t(m)); }

static_assert(uint8_t(fastgltf::CombineMode::Average) == uint8_t(PhysicsCombineMode::Average) && uint8_t(fastgltf::CombineMode::Minimum) == uint8_t(PhysicsCombineMode::Minimum) && uint8_t(fastgltf::CombineMode::Maximum) == uint8_t(PhysicsCombineMode::Maximum) && uint8_t(fastgltf::CombineMode::Multiply) == uint8_t(PhysicsCombineMode::Multiply));
// fastgltf parses an unrecognized combine string as Invalid, which reads as the spec default.
inline PhysicsCombineMode ToCombineMode(fastgltf::CombineMode m) { return m == fastgltf::CombineMode::Invalid ? PhysicsCombineMode::Average : PhysicsCombineMode(uint8_t(m)); }
inline fastgltf::CombineMode FromCombineMode(PhysicsCombineMode m) { return fastgltf::CombineMode(uint8_t(m)); }

// fastgltf numbers animation paths from one.
static_assert(uint8_t(fastgltf::AnimationPath::Translation) == uint8_t(AnimationPath::Translation) + 1 && uint8_t(fastgltf::AnimationPath::Rotation) == uint8_t(AnimationPath::Rotation) + 1 && uint8_t(fastgltf::AnimationPath::Scale) == uint8_t(AnimationPath::Scale) + 1 && uint8_t(fastgltf::AnimationPath::Weights) == uint8_t(AnimationPath::Weights) + 1);
inline AnimationPath ToPath(fastgltf::AnimationPath p) { return AnimationPath(uint8_t(p) - 1); }
inline fastgltf::AnimationPath FromPath(AnimationPath p) { return fastgltf::AnimationPath(uint8_t(p) + 1); }

inline AnimationInterpolation ToInterp(fastgltf::AnimationInterpolation i) {
    switch (i) {
        case fastgltf::AnimationInterpolation::Step: return AnimationInterpolation::Step;
        case fastgltf::AnimationInterpolation::Linear: return AnimationInterpolation::Linear;
        case fastgltf::AnimationInterpolation::CubicSpline: return AnimationInterpolation::CubicSpline;
    }
}
inline fastgltf::AnimationInterpolation FromInterp(AnimationInterpolation i) {
    switch (i) {
        case AnimationInterpolation::Step: return fastgltf::AnimationInterpolation::Step;
        case AnimationInterpolation::Linear: return fastgltf::AnimationInterpolation::Linear;
        case AnimationInterpolation::CubicSpline: return fastgltf::AnimationInterpolation::CubicSpline;
    }
}

inline Filter ToFilter(fastgltf::Filter f) {
    switch (f) {
        case fastgltf::Filter::Nearest: return Filter::Nearest;
        case fastgltf::Filter::Linear: return Filter::Linear;
        case fastgltf::Filter::NearestMipMapNearest: return Filter::NearestMipMapNearest;
        case fastgltf::Filter::LinearMipMapNearest: return Filter::LinearMipMapNearest;
        case fastgltf::Filter::NearestMipMapLinear: return Filter::NearestMipMapLinear;
        case fastgltf::Filter::LinearMipMapLinear: return Filter::LinearMipMapLinear;
    }
}
inline fastgltf::Filter FromFilter(Filter f) {
    switch (f) {
        case Filter::Nearest: return fastgltf::Filter::Nearest;
        case Filter::Linear: return fastgltf::Filter::Linear;
        case Filter::NearestMipMapNearest: return fastgltf::Filter::NearestMipMapNearest;
        case Filter::LinearMipMapNearest: return fastgltf::Filter::LinearMipMapNearest;
        case Filter::NearestMipMapLinear: return fastgltf::Filter::NearestMipMapLinear;
        case Filter::LinearMipMapLinear: return fastgltf::Filter::LinearMipMapLinear;
    }
}
inline Wrap ToWrap(fastgltf::Wrap w) {
    switch (w) {
        case fastgltf::Wrap::ClampToEdge: return Wrap::ClampToEdge;
        case fastgltf::Wrap::MirroredRepeat: return Wrap::MirroredRepeat;
        case fastgltf::Wrap::Repeat: return Wrap::Repeat;
    }
}
inline fastgltf::Wrap FromWrap(Wrap w) {
    switch (w) {
        case Wrap::ClampToEdge: return fastgltf::Wrap::ClampToEdge;
        case Wrap::MirroredRepeat: return fastgltf::Wrap::MirroredRepeat;
        case Wrap::Repeat: return fastgltf::Wrap::Repeat;
    }
}

// Identifies an encoded image from its magic bytes.
inline MimeType SniffMimeType(std::span<const std::byte> bytes) {
    const auto *u8 = reinterpret_cast<const uint8_t *>(bytes.data());
    if (bytes.size() >= 4 && u8[0] == 0x89 && u8[1] == 0x50 && u8[2] == 0x4E && u8[3] == 0x47) return MimeType::PNG;
    if (bytes.size() >= 3 && u8[0] == 0xFF && u8[1] == 0xD8 && u8[2] == 0xFF) return MimeType::JPEG;
    if (bytes.size() >= 12 && std::memcmp(bytes.data(), "RIFF", 4) == 0 && std::memcmp(bytes.data() + 8, "WEBP", 4) == 0) return MimeType::WEBP;
    static constexpr uint8_t Ktx2Magic[12]{0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
    if (bytes.size() >= 12 && std::memcmp(bytes.data(), Ktx2Magic, 12) == 0) return MimeType::KTX2;
    return MimeType::None;
}

// Top-level material texture slots (PBRMaterial accessor, color space, glTF label), ordered to match MaterialTextureSlot.
struct MaterialTextureSlotInfo {
    ::TextureInfo &(*Get)(PBRMaterial &);
    TextureColorSpace ColorSpace;
    std::string_view Label;
};
constexpr std::array<MaterialTextureSlotInfo, MTS_Count> MaterialTextureSlots{{
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.BaseColorTexture; }, TextureColorSpace::Srgb, "baseColor"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.MetallicRoughnessTexture; }, TextureColorSpace::Linear, "metallicRoughness"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.NormalTexture; }, TextureColorSpace::Linear, "normal"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.OcclusionTexture; }, TextureColorSpace::Linear, "occlusion"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.EmissiveTexture; }, TextureColorSpace::Srgb, "emissive"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Specular.Texture; }, TextureColorSpace::Linear, "specular"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Specular.ColorTexture; }, TextureColorSpace::Srgb, "specularColor"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Sheen.ColorTexture; }, TextureColorSpace::Srgb, "sheenColor"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Sheen.RoughnessTexture; }, TextureColorSpace::Linear, "sheenRoughness"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Transmission.Texture; }, TextureColorSpace::Linear, "transmission"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.DiffuseTransmission.Texture; }, TextureColorSpace::Linear, "diffuseTransmission"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.DiffuseTransmission.ColorTexture; }, TextureColorSpace::Srgb, "diffuseTransmissionColor"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Volume.ThicknessTexture; }, TextureColorSpace::Linear, "thickness"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Clearcoat.Texture; }, TextureColorSpace::Linear, "clearcoat"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Clearcoat.RoughnessTexture; }, TextureColorSpace::Linear, "clearcoatRoughness"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Clearcoat.NormalTexture; }, TextureColorSpace::Linear, "clearcoatNormal"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Anisotropy.Texture; }, TextureColorSpace::Linear, "anisotropy"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Iridescence.Texture; }, TextureColorSpace::Linear, "iridescence"},
    {[](PBRMaterial &m) -> ::TextureInfo & { return m.Iridescence.ThicknessTexture; }, TextureColorSpace::Linear, "iridescenceThickness"},
}};

constexpr double Ln1000 = 3 * std::numbers::ln10;

} // namespace gltf::detail
