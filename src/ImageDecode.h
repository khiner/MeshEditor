#pragma once

#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

struct DecodedImage {
    std::vector<std::byte> Pixels;
    uint32_t Width{0}, Height{0};
};

struct DecodedImageF32 {
    std::vector<float> Pixels;
    uint32_t Width{0}, Height{0};
};

std::expected<DecodedImage, std::string> DecodeImageRgba8(std::span<const std::byte> encoded, std::string_view image_name);
std::expected<DecodedImageF32, std::string> DecodeImageRgba32f(std::span<const std::byte> encoded, std::string_view image_name);
std::expected<DecodedImageF32, std::string> DecodeImageFileRgba32f(const std::filesystem::path &, std::string_view image_name);
