#include "File.h"

#include <fcntl.h>
#include <format>
#include <fstream>
#include <sys/file.h>
#include <unistd.h>
#include <utility>

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

DirectoryLock::DirectoryLock(const std::filesystem::path &path) {
    Fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (Fd >= 0 && ::flock(Fd, LOCK_EX | LOCK_NB) != 0) ::close(std::exchange(Fd, -1));
}
DirectoryLock::DirectoryLock(DirectoryLock &&other) noexcept : Fd(std::exchange(other.Fd, -1)) {}
DirectoryLock &DirectoryLock::operator=(DirectoryLock &&other) noexcept {
    if (Fd >= 0) ::close(Fd);
    Fd = std::exchange(other.Fd, -1);
    return *this;
}
DirectoryLock::~DirectoryLock() {
    if (Fd >= 0) ::close(Fd);
}

TemporaryDirectory::TemporaryDirectory(const std::filesystem::path &parent) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    auto name = (parent / "MeshEditor.XXXXXX").string();
    if (!ec && ::mkdtemp(name.data())) Path = name;
}
TemporaryDirectory::~TemporaryDirectory() {
    std::error_code ec;
    if (!Path.empty()) std::filesystem::remove_all(Path, ec);
}

std::expected<std::vector<std::byte>, std::string> Read(const std::filesystem::path &path) { return ReadInto<std::vector<std::byte>>(path); }
std::expected<std::string, std::string> ReadAsString(const std::filesystem::path &path) { return ReadInto<std::string>(path); }

std::expected<void, std::string> WriteAtomic(const std::filesystem::path &path, const std::function<bool(std::ostream &)> &write) {
    auto temporary = path.string() + ".tmp.XXXXXX";
    const int fd = ::mkstemp(temporary.data());
    if (fd < 0) return std::unexpected{std::format("Failed to create temporary file for '{}'.", path.string())};
    ::close(fd);
    struct Cleanup {
        const std::string &Path;
        ~Cleanup() {
            std::error_code ec;
            std::filesystem::remove(Path, ec);
        }
    } cleanup{temporary};
    std::ofstream out{temporary, std::ios::binary | std::ios::trunc};
    const bool written = out && write(out);
    out.close();
    std::error_code ec;
    if (written && out) {
        std::filesystem::rename(temporary, path, ec);
        if (!ec) return {};
    }
    const auto message = ec ? std::format("Failed to replace '{}': {}", path.string(), ec.message()) : std::format("Failed to write '{}'.", path.string());
    return std::unexpected{message};
}

std::expected<void, std::string> WriteAtomic(const std::filesystem::path &path, std::span<const std::byte> bytes) {
    return WriteAtomic(path, [&](auto &out) { return bool(out.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()))); });
}
} // namespace File
