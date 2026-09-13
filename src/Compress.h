#pragma once

#include <filesystem>
#include <optional>
#include <span>
#include <vector>

bool Compress(const std::filesystem::path &src, const std::filesystem::path &dst, std::span<const std::byte> metadata = {});
std::optional<std::vector<std::byte>> ReadArchiveMetadata(const std::filesystem::path &);
bool Decompress(const std::filesystem::path &src, const std::filesystem::path &dst);
