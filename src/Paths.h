#pragma once

#include <filesystem>

namespace Paths {
// `base` is the directory containing the running executable.
// Must be called once at startup before any other `Paths` accessor.
void Init(std::filesystem::path base);

const std::filesystem::path &Base();
const std::filesystem::path &Res();
const std::filesystem::path &Shaders();
} // namespace Paths
