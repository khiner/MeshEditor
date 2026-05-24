#pragma once

#include <cstdint>

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
} // namespace gltf
