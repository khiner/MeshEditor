#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "numeric/vec3.h"

namespace gltf {
enum class MimeType : uint8_t {
    None,
    JPEG,
    PNG,
    KTX2,
    DDS,
    GltfBuffer,
    OctetStream,
    WEBP,
};

struct Image {
    std::vector<std::byte> Bytes;
    MimeType MimeType;
    std::string Name;
};

struct ImageBasedLight {
    std::vector<std::array<uint32_t, 6>> SpecularImageIndicesByMip; // Mip-major; face order: +X, -X, +Y, -Y, +Z, -Z.
    std::optional<std::array<vec3, 9>> IrradianceCoefficients; // L00, L1-1, L10, L11, L2-2, L2-1, L20, L21, L22.
    float Intensity{1.f};
    std::string Name;
};
} // namespace gltf