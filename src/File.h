#pragma once

#include <expected>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <span>
#include <string>
#include <vector>

namespace File {
std::expected<std::vector<std::byte>, std::string> Read(const std::filesystem::path &);
std::expected<std::string, std::string> ReadAsString(const std::filesystem::path &);
// Write beside the destination, then replace it only after writing and closing succeed.
std::expected<void, std::string> WriteAtomic(const std::filesystem::path &, const std::function<bool(std::ostream &)> &write);
std::expected<void, std::string> WriteAtomic(const std::filesystem::path &, std::span<const std::byte>);
} // namespace File
