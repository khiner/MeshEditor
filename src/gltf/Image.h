#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "numeric/quat.h"
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
    // Encoded source bytes. Retained for embedded sources (data URI / bufferView) so save can
    // passthrough byte-identically. External-URI images drop Bytes after upload — SourceAbsPath
    // is the persistence and is re-read at save time.
    std::vector<std::byte> Bytes;
    MimeType MimeType;
    std::string Name, Uri{};
    std::string SourceAbsPath{}; // resolved absolute path; only set when Uri is non-empty.
    bool SourceDataUri{}, SourceHadMimeType{};
    // Flip true when the GPU texture for this image is mutated; SaveGltf then re-encodes
    // from the GPU pixels instead of using Bytes / re-reading the source file.
    bool IsDirty{};
};

struct ImageBasedLight {
    std::vector<std::array<uint32_t, 6>> SpecularImageIndicesByMip{}; // Mip-major; face order: +X, -X, +Y, -Y, +Z, -Z.
    std::optional<std::array<vec3, 9>> IrradianceCoefficients{}; // L00, L1-1, L10, L11, L2-2, L2-1, L20, L21, L22.
    quat Rotation{1, 0, 0, 0}; // Identity quaternion (w, x, y, z).
    uint32_t SpecularImageSize{0}; // Pixel dimension of the highest-resolution specular mip (0 = unspecified).
    float Intensity{1.f};
    std::string Name{};
};
} // namespace gltf