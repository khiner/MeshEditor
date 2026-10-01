#pragma once

#include "state/Entity.h"

#include <expected>
#include <filesystem>
#include <span>
#include <string>

namespace project {
// External files use repository-relative or absolute paths, and generated project files use `asset:/` paths.
struct Assets {
    static constexpr const char *DirectoryName{"assets"}; // Within the project directory.
    std::filesystem::path Directory;
    // A reference is a nonempty relative path.
    static bool IsReference(const std::filesystem::path &);
    std::filesystem::path Resolve(const std::filesystem::path &) const;
    // Reference an absolute path under the asset directory or the repository root, and return any other path unchanged.
    std::filesystem::path Reference(const std::filesystem::path &) const;
    std::expected<std::filesystem::path, std::string> Store(std::string_view name, std::span<const std::byte>);
};

// Resolve references and return other paths unchanged.
std::filesystem::path ResolveAsset(const state::Scene &, const std::filesystem::path &);
std::filesystem::path AssetReference(const state::Scene &, const std::filesystem::path &);
} // namespace project
