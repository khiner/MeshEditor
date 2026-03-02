#pragma once

#include <filesystem>
#include <fstream>

namespace File {
inline std::string Read(const std::filesystem::path path) {
    std::ifstream f{path, std::ios::in | std::ios::binary};
    const auto size = std::filesystem::file_size(path);
    std::string result(size, '\0');
    f.read(result.data(), size);
    return result;
}
} // namespace File
