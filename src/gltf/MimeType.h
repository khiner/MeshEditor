#pragma once

#include <cstdint>
#include <string_view>

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

// The media type string, empty for None.
constexpr std::string_view MimeTypeString(MimeType m) {
    switch (m) {
        case MimeType::None: return "";
        case MimeType::JPEG: return "image/jpeg";
        case MimeType::PNG: return "image/png";
        case MimeType::KTX2: return "image/ktx2";
        case MimeType::DDS: return "image/vnd-ms.dds";
        case MimeType::GltfBuffer: return "model/gltf-buffer";
        case MimeType::OctetStream: return "application/octet-stream";
        case MimeType::WEBP: return "image/webp";
    }
    return "";
}
} // namespace gltf
