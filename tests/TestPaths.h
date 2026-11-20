#pragma once

#include <filesystem>
#include <atomic>
#include <unistd.h>

inline std::filesystem::path ShadersDir(const char *argv0) {
    return std::filesystem::weakly_canonical(std::filesystem::path{argv0}).parent_path() / "shaders";
}

struct TestDir {
    std::filesystem::path Path;
    explicit TestDir(const char *name) : Path(std::filesystem::temp_directory_path() / name) {
        static std::atomic_uint32_t next{};
        Path += "." + std::to_string(getpid()) + "." + std::to_string(next.fetch_add(1));
        std::filesystem::remove_all(Path);
    }
    ~TestDir() { std::filesystem::remove_all(Path); }
    operator const std::filesystem::path &() const { return Path; }
};
