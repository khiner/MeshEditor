#pragma once

#include <array>
#include <cstdint>

enum class MaterialAlphaMode : uint32_t {
    Opaque = 0u,
    Mask = 1u,
    Blend = 2u,
};

inline constexpr uint32_t MaterialAlphaOpaque = uint32_t(MaterialAlphaMode::Opaque);
inline constexpr uint32_t MaterialAlphaMask = uint32_t(MaterialAlphaMode::Mask);
inline constexpr uint32_t MaterialAlphaBlend = uint32_t(MaterialAlphaMode::Blend);
inline constexpr std::array<const char *, 3> MaterialAlphaModeLabels{"Opaque", "Mask", "Blend"};

constexpr uint32_t ToMaterialAlphaModeValue(MaterialAlphaMode mode) { return uint32_t(mode); }
