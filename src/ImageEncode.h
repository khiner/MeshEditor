#pragma once

#include "gltf/Image.h"

#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// Encode tightly-packed RGBA8 pixels to a glTF-supported container (PNG / JPEG).

std::expected<std::vector<std::byte>, std::string> EncodeImagePngRgba8(std::span<const std::byte> rgba8, uint32_t width, uint32_t height, std::string_view image_name);
std::expected<std::vector<std::byte>, std::string> EncodeImageJpegRgba8(std::span<const std::byte> rgba8, uint32_t width, uint32_t height, int quality, std::string_view image_name);
// PNG is lossless. JPEG honors `quality` (1–100).
// WebP / KTX2 / DDS aren't supported — returns an error so callers can decide on a fallback.
std::expected<std::vector<std::byte>, std::string> EncodeImageRgba8ForMime(gltf::MimeType, std::span<const std::byte> rgba8, uint32_t width, uint32_t height, int jpeg_quality, std::string_view image_name);
