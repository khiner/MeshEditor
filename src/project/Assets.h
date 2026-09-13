#pragma once

#include <entt/entity/fwd.hpp>

#include <expected>
#include <filesystem>
#include <span>
#include <string>

namespace project {
// Project-relative references to immutable files.
struct Assets {
    std::filesystem::path Directory;
    static bool IsReference(const std::filesystem::path &);
    std::filesystem::path Resolve(const std::filesystem::path &) const;
    std::filesystem::path Reference(const std::filesystem::path &) const;
    std::expected<std::filesystem::path, std::string> Store(const std::filesystem::path &);
    std::expected<std::filesystem::path, std::string> Store(std::string_view name, std::span<const std::byte>);
};

// Resolve project asset references and return other paths unchanged.
std::filesystem::path ResolveAsset(const entt::registry &, const std::filesystem::path &);
std::filesystem::path AssetReference(const entt::registry &, const std::filesystem::path &);
} // namespace project
