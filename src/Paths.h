#pragma once

#include <filesystem>

namespace Paths {
// Call once at startup before using any Paths accessor.
// Use the read-only executable directory as base and writable per-user storage as user_data.
void Init(std::filesystem::path base, std::filesystem::path user_data);

const std::filesystem::path &Base();
const std::filesystem::path &Res();
// The parent of `Res()` after resolving links, which is the repository root in a build.
const std::filesystem::path &Repo();
const std::filesystem::path &Shaders();
const std::filesystem::path &UserData();

const std::filesystem::path &Project();
void SetProject(std::filesystem::path dir);
} // namespace Paths
