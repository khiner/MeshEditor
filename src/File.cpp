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

std::expected<void, std::string> WriteAtomic(const std::filesystem::path &path, const std::function<bool(std::ostream &)> &write) {
    auto temporary = path;
    temporary += ".tmp";
    std::ofstream out{temporary, std::ios::binary | std::ios::trunc};
    if (!out) return std::unexpected{std::format("Failed to open '{}' for writing.", temporary.string())};
    const bool written = write(out);
    out.close();
    std::error_code ec;
    if (written && out) {
        std::filesystem::rename(temporary, path, ec);
        if (!ec) return {};
    }
    const auto message = ec ? std::format("Failed to replace '{}': {}", path.string(), ec.message()) : std::format("Failed to write '{}'.", path.string());
    std::filesystem::remove(temporary, ec);
    return std::unexpected{message};
}

std::expected<void, std::string> WriteAtomic(const std::filesystem::path &path, std::span<const std::byte> bytes) {
    return WriteAtomic(path, [&](auto &out) { return bool(out.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()))); });
}
} // namespace File
