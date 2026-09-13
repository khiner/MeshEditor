#pragma once

#include <filesystem>

inline std::filesystem::path ShadersDir(const char *argv0) {
    return std::filesystem::weakly_canonical(std::filesystem::path{argv0}).parent_path() / "shaders";
}

struct TestDir {
    std::filesystem::path Path;
    explicit TestDir(const char *name) : Path(std::filesystem::temp_directory_path() / name) { std::filesystem::remove_all(Path); }
    ~TestDir() { std::filesystem::remove_all(Path); }
    operator const std::filesystem::path &() const { return Path; }
};
