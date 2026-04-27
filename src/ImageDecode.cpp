#include "ImageDecode.h"

#include <format>

// Decode imported image bytes (PNG/JPEG/HDR/etc.) to fixed RGBA layouts for GPU uploads.
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "../lib/lunasvg/plutovg/source/plutovg-stb-image.h"

namespace {
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
} // namespace

std::expected<DecodedImage, std::string> DecodeImageRgba8(std::span<const std::byte> encoded, std::string_view image_name) {
    if (encoded.empty()) return std::unexpected{std::format("Image '{}' has no encoded bytes.", image_name)};

    int width{0}, height{0}, channels{0};
    stbi_uc *pixels = stbi_load_from_memory(reinterpret_cast<const stbi_uc *>(encoded.data()), encoded.size(), &width, &height, &channels, 4);
    if (!pixels) return std::unexpected{std::format("Failed to decode image '{}': {}", image_name, StbiFailureReasonOrUnknown())};

    if (auto valid = ValidateDimensions(width, height, image_name); !valid) {
        stbi_image_free(pixels);
        return std::unexpected{std::move(valid.error())};
    }

    DecodedImage decoded{.Pixels = std::vector<std::byte>(size_t(width) * size_t(height) * 4u), .Width = uint32_t(width), .Height = uint32_t(height)};
    std::memcpy(decoded.Pixels.data(), pixels, decoded.Pixels.size());
    stbi_image_free(pixels);
    return decoded;
}

std::expected<DecodedImageF32, std::string> DecodeImageRgba32f(std::span<const std::byte> encoded, std::string_view image_name) {
    if (encoded.empty()) return std::unexpected{std::format("Image '{}' has no encoded bytes.", image_name)};

    int width{0}, height{0}, channels{0};
    float *pixels = stbi_loadf_from_memory(reinterpret_cast<const stbi_uc *>(encoded.data()), encoded.size(), &width, &height, &channels, 4);
    if (!pixels) return std::unexpected{std::format("Failed to decode image '{}' as float RGBA: {}", image_name, StbiFailureReasonOrUnknown())};

    if (auto valid = ValidateDimensions(width, height, image_name); !valid) {
        stbi_image_free(pixels);
        return std::unexpected{std::move(valid.error())};
    }

    DecodedImageF32 decoded{.Pixels = std::vector<float>(size_t(width) * size_t(height) * 4u), .Width = uint32_t(width), .Height = uint32_t(height)};
    std::memcpy(decoded.Pixels.data(), pixels, decoded.Pixels.size() * sizeof(float));
    stbi_image_free(pixels);
    return decoded;
}

std::expected<DecodedImageF32, std::string> DecodeImageFileRgba32f(const std::filesystem::path &path, std::string_view image_name) {
    const auto path_string = path.string();

    int width{0}, height{0}, channels{0};
    float *pixels = stbi_loadf(path_string.c_str(), &width, &height, &channels, 4);
    if (!pixels) return std::unexpected{StbiFailureReasonOrUnknown()};

    if (auto valid = ValidateDimensions(width, height, image_name.empty() ? path_string : std::string{image_name}); !valid) {
        stbi_image_free(pixels);
        return std::unexpected{std::move(valid.error())};
    }

    DecodedImageF32 decoded{.Pixels = std::vector<float>(width * height * 4), .Width = uint32_t(width), .Height = uint32_t(height)};
    std::memcpy(decoded.Pixels.data(), pixels, decoded.Pixels.size() * sizeof(float));
    stbi_image_free(pixels);
    return decoded;
}
