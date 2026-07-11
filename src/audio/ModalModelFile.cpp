#include "ModalModelFile.h"

#include "Paths.h"
#include "action/SerializeGlm.h"

#include <zpp_bits.h>

#include <format>
#include <fstream>

namespace fs = std::filesystem;

namespace {
std::vector<std::byte> Serialize(const ModalModelData &data) {
    std::vector<std::byte> bytes;
    zpp::bits::out archive{bytes};
    // zpp's aggregate reflection mis-encodes const aggregates, so serialize through a non-const ref.
    if (zpp::bits::failure(archive(const_cast<ModalModelData &>(data)))) return {};
    return bytes;
}

std::optional<std::vector<std::byte>> ReadFileBytes(const fs::path &path) {
    std::ifstream in{path, std::ios::binary | std::ios::ate};
    if (!in) return {};
    std::vector<std::byte> bytes(in.tellg());
    in.seekg(0);
    in.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
    if (!in) return {};
    return bytes;
}
} // namespace

fs::path ModalModelsDir() { return Paths::Base() / "modal"; }

fs::path SaveModalModelFile(const ModalModelData &data) {
    const auto bytes = Serialize(data);
    if (bytes.empty()) return {};

    const auto dir = ModalModelsDir();
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) return {};

    const size_t hash = std::hash<std::string_view>{}({reinterpret_cast<const char *>(bytes.data()), bytes.size()});
    // Content-hash naming keeps files write-once: identical content reuses the file, and a hash
    // collision with different content moves to a suffixed name.
    for (uint32_t suffix = 0;; ++suffix) {
        auto name = suffix == 0 ? std::format("{:016x}.modal", hash) : std::format("{:016x}-{}.modal", hash, suffix);
        const auto path = dir / name;
        if (fs::exists(path)) {
            if (ReadFileBytes(path) == bytes) return name;
            continue;
        }
        std::ofstream out{path, std::ios::binary};
        out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        return out ? fs::path{std::move(name)} : fs::path{};
    }
}

std::optional<ModalModelData> LoadModalModelFile(const fs::path &relative) {
    const auto bytes = ReadFileBytes(ModalModelsDir() / relative);
    if (!bytes) return {};

    ModalModelData data;
    if (zpp::bits::failure(zpp::bits::in{*bytes}(data))) return {};
    return data;
}
