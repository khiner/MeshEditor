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

inline uint64_t ExtrasKey(fastgltf::Category cat, size_t idx) { return (uint64_t(uint32_t(cat)) << 32) | uint64_t(idx); }
// Maps enums in either direction and returns `fallback` for unmapped values.
template<typename A, typename B, size_t N>
constexpr B MapEnum(const std::pair<A, B> (&table)[N], A from, B fallback) {
    for (const auto &[a, b] : table) {
        if (a == from) return b;
    }
    return fallback;
}
template<typename A, typename B, size_t N>
constexpr A MapEnumBack(const std::pair<A, B> (&table)[N], B from, A fallback) {
    for (const auto &[a, b] : table) {
        if (b == from) return a;
    }
    return fallback;
}

constexpr std::pair<fastgltf::Filter, Filter> FilterMap[]{
    {fastgltf::Filter::Nearest, Filter::Nearest},
    {fastgltf::Filter::Linear, Filter::Linear},
    {fastgltf::Filter::NearestMipMapNearest, Filter::NearestMipMapNearest},
    {fastgltf::Filter::LinearMipMapNearest, Filter::LinearMipMapNearest},
    {fastgltf::Filter::NearestMipMapLinear, Filter::NearestMipMapLinear},
    {fastgltf::Filter::LinearMipMapLinear, Filter::LinearMipMapLinear},
};
constexpr std::pair<fastgltf::Wrap, Wrap> WrapMap[]{
    {fastgltf::Wrap::ClampToEdge, Wrap::ClampToEdge},
    {fastgltf::Wrap::MirroredRepeat, Wrap::MirroredRepeat},
    {fastgltf::Wrap::Repeat, Wrap::Repeat},
};
constexpr std::pair<fastgltf::MimeType, MimeType> MimeTypeMap[]{
    {fastgltf::MimeType::None, MimeType::None},
    {fastgltf::MimeType::JPEG, MimeType::JPEG},
    {fastgltf::MimeType::PNG, MimeType::PNG},
    {fastgltf::MimeType::KTX2, MimeType::KTX2},
    {fastgltf::MimeType::DDS, MimeType::DDS},
    {fastgltf::MimeType::GltfBuffer, MimeType::GltfBuffer},
    {fastgltf::MimeType::OctetStream, MimeType::OctetStream},
    {fastgltf::MimeType::WEBP, MimeType::WEBP},
};
constexpr std::pair<fastgltf::AlphaMode, MaterialAlphaMode> AlphaModeMap[]{
    {fastgltf::AlphaMode::Opaque, MaterialAlphaMode::Opaque},
    {fastgltf::AlphaMode::Mask, MaterialAlphaMode::Mask},
    {fastgltf::AlphaMode::Blend, MaterialAlphaMode::Blend},
};
constexpr std::pair<fastgltf::AnimationInterpolation, AnimationInterpolation> InterpMap[]{
    {fastgltf::AnimationInterpolation::Step, AnimationInterpolation::Step},
    {fastgltf::AnimationInterpolation::Linear, AnimationInterpolation::Linear},
    {fastgltf::AnimationInterpolation::CubicSpline, AnimationInterpolation::CubicSpline},
};
constexpr std::pair<fastgltf::AnimationPath, AnimationPath> PathMap[]{
    {fastgltf::AnimationPath::Translation, AnimationPath::Translation},
    {fastgltf::AnimationPath::Rotation, AnimationPath::Rotation},
    {fastgltf::AnimationPath::Scale, AnimationPath::Scale},
    {fastgltf::AnimationPath::Weights, AnimationPath::Weights},
};
constexpr std::pair<fastgltf::CombineMode, PhysicsCombineMode> CombineMap[]{
    {fastgltf::CombineMode::Average, PhysicsCombineMode::Average},
    {fastgltf::CombineMode::Minimum, PhysicsCombineMode::Minimum},
    {fastgltf::CombineMode::Maximum, PhysicsCombineMode::Maximum},
    {fastgltf::CombineMode::Multiply, PhysicsCombineMode::Multiply},
};

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
