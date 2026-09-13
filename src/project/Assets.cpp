#include "project/Assets.h"

#include "File.h"
#include "project/store/Hash.h"

#include <entt/entity/registry.hpp>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <format>
#include <memory>

namespace project {
namespace fs = std::filesystem;

bool Assets::IsReference(const fs::path &path) { return path.native().starts_with("asset:/"); }

fs::path Assets::Resolve(const fs::path &path) const {
    return IsReference(path) ? Directory / "assets" / path.native().substr(7) : path;
}

fs::path Assets::Reference(const fs::path &path) const {
    if (Directory.empty() || IsReference(path)) return path;
    const auto relative = fs::absolute(path).lexically_normal().lexically_relative(fs::absolute(Directory / "assets").lexically_normal());
    if (relative.empty() || *relative.begin() == "..") return path;
    return fs::path{"asset:"} / relative;
}

std::expected<fs::path, std::string> Assets::Store(std::string_view name, std::span<const std::byte> bytes) {
    const auto hash = store::HashBytes(bytes);
    const auto reference = fs::path{"asset:"} / std::format("{:016x}{:016x}", hash.A, hash.B) / fs::path{name}.filename();
    const auto path = Resolve(reference);
    std::error_code ec;
    if (fs::is_regular_file(path, ec)) return reference;
    fs::create_directories(path.parent_path(), ec);
    if (ec) return std::unexpected{"Cannot create asset directory: " + ec.message()};
    if (const auto written = File::WriteAtomic(path, bytes); !written) return std::unexpected{written.error()};
    return reference;
}

std::expected<fs::path, std::string> Assets::Store(const fs::path &source) {
    if (IsReference(source)) return source;
    const int fd = ::open(source.c_str(), O_RDONLY);
    if (fd < 0) return std::unexpected{std::format("Cannot open asset '{}': {}", source.string(), std::strerror(errno))};
    struct stat info{};
    const bool valid = ::fstat(fd, &info) == 0 && S_ISREG(info.st_mode);
    void *mapped = valid && info.st_size ? ::mmap(nullptr, size_t(info.st_size), PROT_READ, MAP_PRIVATE, fd, 0) : nullptr;
    ::close(fd);
    if (!valid || mapped == MAP_FAILED) return std::unexpected{std::format("Cannot read asset '{}'.", source.string())};
    const auto unmap = [size = size_t(info.st_size)](void *p) { ::munmap(p, size); };
    std::unique_ptr<void, decltype(unmap)> mapping{mapped, unmap};
    return Store(source.filename().string(), {static_cast<const std::byte *>(mapped), size_t(info.st_size)});
}

fs::path ResolveAsset(const entt::registry &r, const fs::path &path) {
    return Assets::IsReference(path) ? r.ctx().get<const Assets>().Resolve(path) : path;
}

fs::path AssetReference(const entt::registry &r, const fs::path &path) {
    const auto *assets = r.ctx().find<const Assets>();
    return assets ? assets->Reference(path) : path;
}
} // namespace project
