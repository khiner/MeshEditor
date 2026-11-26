#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

#include <Foundation/NSSharedPtr.hpp>

namespace MTL {
class CompileOptions;
}

namespace msl {
// Runtime compile options matching the offline shader build, with each math mode declared in its source.
NS::SharedPtr<MTL::CompileOptions> CompileOptions();
struct Source {
    std::string Text;
    std::vector<std::filesystem::path> Files;
    bool CanLoadBinary(const std::filesystem::path &root, const std::filesystem::path &binary) const;
};

// Returns flattened shader source with `defines` prepended and repeated includes omitted.
Source Load(const std::filesystem::path &root, const std::filesystem::path &relative_path, const std::vector<std::string> &defines = {});
} // namespace msl
