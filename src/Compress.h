#pragma once

#include <filesystem>
#include <optional>
#include <span>
#include <vector>

// Archive the regular files under src, limited to the files at or under the named relative paths when `entries` is nonempty.
bool Compress(const std::filesystem::path &src, const std::filesystem::path &dst, std::span<const std::byte> metadata = {}, std::span<const std::string_view> entries = {});
std::optional<std::vector<std::byte>> ReadArchiveMetadata(const std::filesystem::path &);
bool Decompress(const std::filesystem::path &src, const std::filesystem::path &dst);
