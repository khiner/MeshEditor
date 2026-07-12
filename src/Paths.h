#pragma once

#include <filesystem>

namespace Paths {
// `base` is the (read-only) directory containing the running executable.
// `user_data` is the writable per-user data directory (projects, scratch sessions).
// Must be called once at startup before any other `Paths` accessor.
void Init(std::filesystem::path base, std::filesystem::path user_data);

const std::filesystem::path &Base();
const std::filesystem::path &Res();
const std::filesystem::path &Shaders();
const std::filesystem::path &UserData();

const std::filesystem::path &Project();
void SetProject(std::filesystem::path dir);
} // namespace Paths
