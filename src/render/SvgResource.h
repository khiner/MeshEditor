#pragma once

#include "numeric/vec2.h"

#include <filesystem>

using numeric::vec2;

namespace mtl {
struct Context;
} // namespace mtl

struct SvgResource {
    SvgResource(const mtl::Context &, const std::filesystem::path &);
    ~SvgResource();

    // Draws the icon over [min, max] in the current window's draw list.
    void DrawIcon(vec2 min, vec2 max) const;

private:
    struct Impl;
    std::unique_ptr<Impl> Imp;
};

std::unique_ptr<SvgResource> LoadSvg(const mtl::Context &, const std::filesystem::path &);
