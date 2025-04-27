#pragma once

#include <filesystem>
#include <functional>
#include <memory>

struct SvgResource;
// using CreateSvgResource = void (*)(std::unique_ptr<SvgResource> &, std::filesystem::path);
using CreateSvgResource = std::function<void(std::unique_ptr<SvgResource> &, std::filesystem::path)>;
