#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

// Encode tightly-packed RGBA8 pixels. PNG is lossless. JPEG honors `quality` (1–100).
std::expected<std::vector<std::byte>, std::string> EncodeImagePngRgba8(std::span<const std::byte> rgba8, uint32_t width, uint32_t height, std::string_view name);
std::expected<std::vector<std::byte>, std::string> EncodeImageJpegRgba8(std::span<const std::byte> rgba8, uint32_t width, uint32_t height, int quality, std::string_view name);
