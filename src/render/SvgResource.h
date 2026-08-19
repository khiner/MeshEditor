#pragma once

#include "numeric/vec2.h"

#include <filesystem>
#include <memory>

namespace mtl {
struct Context;
} // namespace mtl

struct SvgResource {
    SvgResource(const mtl::Context &, const std::filesystem::path &);
    ~SvgResource();

    // Draw icon at given size (no interaction)
    void DrawIcon(vec2 size) const;

private:
    struct Impl;
    std::unique_ptr<Impl> Imp;
};

std::unique_ptr<SvgResource> LoadSvg(const mtl::Context &, const std::filesystem::path &);
