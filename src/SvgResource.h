#pragma once

#include "vulkan/Image.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <span>

namespace fs = std::filesystem;

using BitmapToImage = std::function<mvk::ImageResource(std::span<const std::byte> data, uint32_t width, uint32_t height)>;

struct SvgResource {
    SvgResource(vk::Device, BitmapToImage, fs::path);
    ~SvgResource();

    // Returns the clicked link path.
    std::optional<fs::path> Draw();
    // Draw icon at given size (no interaction)
    void DrawIcon(vec2 size) const;

    fs::path Path;

private:
    struct Impl;
    std::unique_ptr<Impl> Imp;
};
