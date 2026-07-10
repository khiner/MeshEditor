#pragma once

#include "numeric/vec2.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>

namespace vk {
class Device;
} // namespace vk
namespace mvk {
struct ImageResource;
} // namespace mvk

using BitmapToImage = std::function<mvk::ImageResource(std::span<const std::byte> data, uint32_t width, uint32_t height)>;

struct SvgResource {
    SvgResource(vk::Device, const BitmapToImage &, const std::filesystem::path &);
    ~SvgResource();

    // Draw icon at given size (no interaction)
    void DrawIcon(vec2 size) const;

private:
    struct Impl;
    std::unique_ptr<Impl> Imp;
};
