#pragma once

#include <filesystem>

inline std::filesystem::path ShadersDir(const char *argv0) {
    return std::filesystem::weakly_canonical(std::filesystem::path{argv0}).parent_path() / "shaders";
}
