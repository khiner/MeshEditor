#pragma once

#include <filesystem>
#include <memory>
#include <optional>

#include "vulkan/Vulkan.h"

namespace fs = std::filesystem;

struct SvgResourceImpl;

struct SvgResource {
    SvgResource(vk::Device, RenderBitmapToImage, fs::path);
    ~SvgResource();

    // Returns the clicked link path.
    std::optional<fs::path> Render();

    fs::path Path;
    std::unique_ptr<SvgResourceImpl> Impl;
};
