#pragma once

#include "Vulkan/Image.h"

#include <filesystem>
#include <memory>
#include <optional>

namespace fs = std::filesystem;

struct SvgResourceImpl;

struct SvgResource {
    SvgResource(vk::Device, mvk::BitmapToImage, fs::path);
    ~SvgResource();

    // Returns the clicked link path.
    std::optional<fs::path> Render();

    fs::path Path;
    std::unique_ptr<SvgResourceImpl> Impl;
};
