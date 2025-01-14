#pragma once

#include <filesystem>
#include <memory>

struct SvgResource;
using CreateSvgResource = void (*)(std::unique_ptr<SvgResource> &, std::filesystem::path);
