#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace msl {
struct Source {
    std::string Text;
    std::vector<std::filesystem::path> Files;
};

// Returns flattened shader source with `defines` prepended and repeated includes omitted.
Source Load(const std::filesystem::path &root, const std::filesystem::path &relative_path, const std::vector<std::string> &defines = {});
} // namespace msl
