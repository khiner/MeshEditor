#pragma once

#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace File {
std::expected<std::vector<std::byte>, std::string> Read(const std::filesystem::path &);
std::expected<std::string, std::string> ReadAsString(const std::filesystem::path &);
} // namespace File
