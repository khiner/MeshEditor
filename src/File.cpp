#include "File.h"

#include <format>
#include <fstream>

namespace File {
namespace {
template<typename T> std::expected<T, std::string> ReadInto(const std::filesystem::path &path) {
    std::ifstream f{path, std::ios::binary | std::ios::ate};
    if (!f) return std::unexpected{std::format("Failed to open '{}'.", path.string())};
    const auto size = f.tellg();
    if (size <= 0) return T{};
    T result(size_t(size), {});
    f.seekg(0);
    f.read(reinterpret_cast<char *>(result.data()), std::streamsize(size));
    if (!f) return std::unexpected{std::format("Failed to read '{}'.", path.string())};
    return result;
}
} // namespace

std::expected<std::vector<std::byte>, std::string> Read(const std::filesystem::path &path) { return ReadInto<std::vector<std::byte>>(path); }
std::expected<std::string, std::string> ReadAsString(const std::filesystem::path &path) { return ReadInto<std::string>(path); }
} // namespace File
