#include "image/ImageDecode.h"

#include <format>

// Decode imported image bytes (PNG/JPEG/WebP/HDR/etc.) to fixed RGBA layouts for GPU uploads.
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <plutovg-stb-image.h>

#include <webp/decode.h>

namespace {
bool IsWebp(std::span<const std::byte> bytes) {
    return bytes.size() >= 12 && std::memcmp(bytes.data(), "RIFF", 4) == 0 && std::memcmp(bytes.data() + 8, "WEBP", 4) == 0;
}

std::expected<DecodedImage, std::string> DecodeWebpRgba8(std::span<const std::byte> encoded, std::string_view image_name) {
    int width{0}, height{0};
    uint8_t *pixels = WebPDecodeRGBA(reinterpret_cast<const uint8_t *>(encoded.data()), encoded.size(), &width, &height);
    if (!pixels) return std::unexpected{std::format("Failed to decode WebP image '{}'.", image_name)};

    DecodedImage decoded{.Pixels = std::vector<std::byte>(size_t(width) * size_t(height) * 4u), .Width = uint32_t(width), .Height = uint32_t(height)};
    std::memcpy(decoded.Pixels.data(), pixels, decoded.Pixels.size());
    WebPFree(pixels);
    return decoded;
}

std::string StbiFailureReasonOrUnknown() {
    const char *reason = stbi_failure_reason();
    return reason ? reason : "unknown stb_image error";
}

std::expected<void, std::string> ValidateDimensions(int width, int height, std::string_view image_name) {
    if (width <= 0 || height <= 0) {
        return std::unexpected{std::format("Image '{}' decoded to invalid dimensions {}x{}.", image_name, width, height)};
    }
    return {};
}

// Run `load` (fills width/height, returns stb-allocated pixels or null), validate dims, copy into Result.
template<typename Result, typename Pixel, typename Load>
std::expected<Result, std::string> DecodeImage(std::string_view image_name, Load &&load) {
    int width{0}, height{0}, channels{0};
    void *pixels = load(&width, &height, &channels);
    if (!pixels) return std::unexpected{std::format("Failed to decode image '{}': {}", image_name, StbiFailureReasonOrUnknown())};

    if (auto valid = ValidateDimensions(width, height, image_name); !valid) {
        stbi_image_free(pixels);
        return std::unexpected{std::move(valid.error())};
    }

    Result decoded{.Pixels = std::vector<Pixel>(size_t(width) * size_t(height) * 4u), .Width = uint32_t(width), .Height = uint32_t(height)};
    std::memcpy(decoded.Pixels.data(), pixels, decoded.Pixels.size() * sizeof(Pixel));
    stbi_image_free(pixels);
    return decoded;
}
} // namespace

std::expected<DecodedImage, std::string> DecodeImageRgba8(std::span<const std::byte> encoded, std::string_view image_name) {
    if (encoded.empty()) return std::unexpected{std::format("Image '{}' has no encoded bytes.", image_name)};
    if (IsWebp(encoded)) return DecodeWebpRgba8(encoded, image_name);
    return DecodeImage<DecodedImage, std::byte>(image_name, [&](int *w, int *h, int *c) -> void * {
        return stbi_load_from_memory(reinterpret_cast<const stbi_uc *>(encoded.data()), encoded.size(), w, h, c, 4);
    });
}

std::expected<DecodedImageF32, std::string> DecodeImageRgba32f(std::span<const std::byte> encoded, std::string_view image_name) {
    if (encoded.empty()) return std::unexpected{std::format("Image '{}' has no encoded bytes.", image_name)};
    return DecodeImage<DecodedImageF32, float>(image_name, [&](int *w, int *h, int *c) -> void * {
        return stbi_loadf_from_memory(reinterpret_cast<const stbi_uc *>(encoded.data()), encoded.size(), w, h, c, 4);
    });
}

std::expected<DecodedImageF32, std::string> DecodeImageFileRgba32f(const std::filesystem::path &path, std::string_view image_name) {
    const auto path_string = path.string();
    const std::string name = image_name.empty() ? path_string : std::string{image_name};
    return DecodeImage<DecodedImageF32, float>(name, [&](int *w, int *h, int *c) -> void * {
        return stbi_loadf(path_string.c_str(), w, h, c, 4);
    });
}
