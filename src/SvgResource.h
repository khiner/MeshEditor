#pragma once

#include <filesystem>
#include <memory>
#include <optional>

namespace fs = std::filesystem;

struct SvgResourceImpl;
struct VulkanContext;

struct SvgResource {
    SvgResource(const VulkanContext &, fs::path);
    ~SvgResource();

    // Returns the clicked link path.
    std::optional<fs::path> Render();

    fs::path Path;
    std::unique_ptr<SvgResourceImpl> Impl;
};
