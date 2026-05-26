#include "image/ImageEncode.h"

#include <format>

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <plutovg-stb-image-write.h>

namespace {
void AppendToVector(void *ctx, void *data, int size) {
    if (size <= 0) return;
    auto *out = static_cast<std::vector<std::byte> *>(ctx);
    const auto offset = out->size();
    out->resize(offset + size);
    std::memcpy(out->data() + offset, data, size);
}

std::expected<void, std::string> ValidateInput(std::span<const std::byte> rgba8, uint32_t width, uint32_t height, std::string_view name) {
    if (width == 0 || height == 0) {
        return std::unexpected{std::format("Image '{}' has zero dimension {}x{}.", name, width, height)};
    }
    if (const size_t expected = width * height * 4; rgba8.size() != expected) {
        return std::unexpected{std::format("Image '{}' RGBA8 buffer size {} doesn't match {}x{}x4 = {}.", name, rgba8.size(), width, height, expected)};
    }
    return {};
}
} // namespace

std::expected<std::vector<std::byte>, std::string>
EncodeImagePngRgba8(std::span<const std::byte> rgba8, uint32_t width, uint32_t height, std::string_view name) {
    if (auto valid = ValidateInput(rgba8, width, height, name); !valid) return std::unexpected{std::move(valid.error())};

    std::vector<std::byte> out;
    out.reserve(width * height);
    if (!stbi_write_png_to_func(&AppendToVector, &out, width, height, 4, rgba8.data(), width * 4)) {
        return std::unexpected{std::format("Failed to PNG-encode image '{}'.", name)};
    }
    return out;
}

std::expected<std::vector<std::byte>, std::string>
EncodeImageJpegRgba8(std::span<const std::byte> rgba8, uint32_t width, uint32_t height, int quality, std::string_view name) {
    if (auto valid = ValidateInput(rgba8, width, height, name); !valid) return std::unexpected{std::move(valid.error())};

    std::vector<std::byte> out;
    out.reserve(width * height / 4);
    if (!stbi_write_jpg_to_func(&AppendToVector, &out, width, height, 4, rgba8.data(), std::clamp(quality, 1, 100))) {
        return std::unexpected{std::format("Failed to JPEG-encode image '{}'.", name)};
    }
    return out;
}
