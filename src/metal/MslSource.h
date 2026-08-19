#pragma once

#include <filesystem>
#include <string>
#include <vector>

// Resolves shader includes into the self-contained source Metal expects.
namespace msl {
struct Source {
    std::string Text;
    std::vector<std::filesystem::path> Files;
};

// Repeated includes are skipped; `defines` are emitted before the flattened source.
Source Load(const std::filesystem::path &root, const std::filesystem::path &relative_path, const std::vector<std::string> &defines = {});
} // namespace msl
