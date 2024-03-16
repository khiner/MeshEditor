#pragma once

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace File {
inline std::string Read(const fs::path path) {
    std::ifstream f(path, std::ios::in | std::ios::binary);
    const auto size = fs::file_size(path);
    std::string result(size, '\0');
    f.read(result.data(), size);
    return result;
}
} // namespace File
