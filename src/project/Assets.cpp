#include "project/Assets.h"

#include "File.h"
#include "Paths.h"
#include "project/store/Hash.h"

#include "state/Scene.h"

#include <format>

namespace project {
namespace fs = std::filesystem;

bool Assets::IsReference(const fs::path &path) { return !path.empty() && path.is_relative(); }

fs::path Assets::Resolve(const fs::path &path) const {
    if (!IsReference(path)) return path;
    return path.native().starts_with("asset:/") ? Directory / DirectoryName / path.native().substr(7) : Paths::Repo() / path;
}

fs::path Assets::Reference(const fs::path &path) const {
    if (IsReference(path)) return path;
    if (!Directory.empty()) {
        const auto relative = fs::absolute(path).lexically_normal().lexically_relative(fs::absolute(Directory / DirectoryName).lexically_normal());
        if (!relative.empty() && *relative.begin() != "..") return fs::path{"asset:"} / relative;
    }
    std::error_code ec;
    const auto relative = fs::weakly_canonical(path, ec).lexically_relative(Paths::Repo());
    return ec || Paths::Repo().empty() || relative.empty() || *relative.begin() == ".." ? path : relative;
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

fs::path ResolveAsset(const state::Scene &r, const fs::path &path) {
    return Assets::IsReference(path) ? r.Context.get<const Assets>().Resolve(path) : path;
}

fs::path AssetReference(const state::Scene &r, const fs::path &path) {
    const auto *assets = r.Context.find<const Assets>();
    return assets ? assets->Reference(path) : path;
}
} // namespace project
