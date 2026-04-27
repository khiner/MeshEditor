#include "ImageEncode.h"

#include <algorithm>
#include <cstring>
#include <format>

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/lunasvg/plutovg/source/plutovg-stb-image-write.h"

namespace {
void AppendToVector(void *ctx, void *data, int size) {
    if (size <= 0) return;
    auto *out = static_cast<std::vector<std::byte> *>(ctx);
    const auto offset = out->size();
    out->resize(offset + size_t(size));
    std::memcpy(out->data() + offset, data, size_t(size));
}

std::expected<void, std::string> ValidateInput(std::span<const std::byte> rgba8, uint32_t width, uint32_t height, std::string_view image_name) {
    if (width == 0 || height == 0) {
        return std::unexpected{std::format("Image '{}' has zero dimension {}x{}.", image_name, width, height)};
    }
    const size_t expected = size_t(width) * size_t(height) * 4u;
    if (rgba8.size() != expected) {
        return std::unexpected{std::format("Image '{}' RGBA8 buffer size {} doesn't match {}x{}x4 = {}.", image_name, rgba8.size(), width, height, expected)};
    }
    return {};
}
} // namespace

std::expected<std::vector<std::byte>, std::string>
EncodeImagePngRgba8(std::span<const std::byte> rgba8, uint32_t width, uint32_t height, std::string_view image_name) {
    if (auto valid = ValidateInput(rgba8, width, height, image_name); !valid) return std::unexpected{std::move(valid.error())};

    std::vector<std::byte> out;
    out.reserve(size_t(width) * size_t(height));
    const int stride = int(width) * 4;
    if (!stbi_write_png_to_func(&AppendToVector, &out, int(width), int(height), 4, rgba8.data(), stride)) {
        return std::unexpected{std::format("Failed to PNG-encode image '{}'.", image_name)};
    }
    return out;
}

std::expected<std::vector<std::byte>, std::string>
EncodeImageJpegRgba8(std::span<const std::byte> rgba8, uint32_t width, uint32_t height, int quality, std::string_view image_name) {
    if (auto valid = ValidateInput(rgba8, width, height, image_name); !valid) return std::unexpected{std::move(valid.error())};

    const int q = std::clamp(quality, 1, 100);
    std::vector<std::byte> out;
    out.reserve(size_t(width) * size_t(height) / 4u);
    if (!stbi_write_jpg_to_func(&AppendToVector, &out, int(width), int(height), 4, rgba8.data(), q)) {
        return std::unexpected{std::format("Failed to JPEG-encode image '{}'.", image_name)};
    }
    return out;
}

std::expected<std::vector<std::byte>, std::string>
EncodeImageRgba8ForMime(gltf::MimeType mime, std::span<const std::byte> rgba8, uint32_t width, uint32_t height, int jpeg_quality, std::string_view image_name) {
    using enum gltf::MimeType;
    switch (mime) {
        case PNG: return EncodeImagePngRgba8(rgba8, width, height, image_name);
        case JPEG: return EncodeImageJpegRgba8(rgba8, width, height, jpeg_quality, image_name);
        case WEBP: return std::unexpected{std::format("WebP encoding not supported for image '{}'; caller should fall back to PNG.", image_name)};
        case KTX2: return std::unexpected{std::format("KTX2 encoding not supported for image '{}' (no basisu encoder vendored).", image_name)};
        case DDS: return std::unexpected{std::format("DDS encoding not supported for image '{}'.", image_name)};
        case GltfBuffer:
        case OctetStream:
        case None: return std::unexpected{std::format("Unrecognized mime type for image '{}'.", image_name)};
    }
    return std::unexpected{std::format("Unhandled mime type for image '{}'.", image_name)};
}
