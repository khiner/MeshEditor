#include "Compress.h"
#include "File.h"

#include <zstd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
constexpr int CompressionLevel{5};
constexpr size_t ChunkSize{1 << 20}; // 1 MiB streaming buffer.
constexpr uint32_t MetadataMagic = 0x184D2A50; // Zstd skippable frame.

// Encode each archive entry as [uint32 path length][path][uint64 data length][data] in one zstd stream.

// Compress `in` to `out`; use ZSTD_e_end for the final flush.
bool Feed(ZSTD_CCtx *cctx, std::ostream &out, std::vector<char> &buf, ZSTD_inBuffer in, ZSTD_EndDirective mode) {
    for (bool done = false; !done;) {
        ZSTD_outBuffer o{buf.data(), buf.size(), 0};
        const size_t remaining = ZSTD_compressStream2(cctx, &o, &in, mode);
        if (ZSTD_isError(remaining)) return false;

        out.write(buf.data(), std::streamsize(o.pos));
        if (!out) return false;

        done = mode == ZSTD_e_end ? remaining == 0 : in.pos == in.size;
    }
    return true;
}
bool CompressToStream(const fs::path &src, std::ostream &out) {
    std::error_code ec;
    const std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)> cctx{ZSTD_createCCtx(), ZSTD_freeCCtx};
    if (!cctx) return false;

    ZSTD_CCtx_setParameter(cctx.get(), ZSTD_c_compressionLevel, CompressionLevel);

    std::vector<char> out_buf(ZSTD_CStreamOutSize()), chunk(ChunkSize);
    for (fs::recursive_directory_iterator it{src, ec}, end; !ec && it != end; it.increment(ec)) {
        const bool regular = it->is_regular_file(ec);
        if (ec) return false;
        if (!regular) continue;
        const auto rel = it->path().lexically_relative(src).generic_string();
        const auto data_len = uint64_t(fs::file_size(it->path(), ec));
        if (ec) return false;

        std::ifstream in{it->path(), std::ios::binary};
        if (!in) return false;

        const uint32_t path_len = uint32_t(rel.size());
        if (!Feed(cctx.get(), out, out_buf, {&path_len, sizeof path_len, 0}, ZSTD_e_continue) ||
            !Feed(cctx.get(), out, out_buf, {rel.data(), rel.size(), 0}, ZSTD_e_continue) ||
            !Feed(cctx.get(), out, out_buf, {&data_len, sizeof data_len, 0}, ZSTD_e_continue)) return false;

        for (uint64_t left = data_len; left > 0;) {
            const std::streamsize n = std::streamsize(std::min<uint64_t>(left, chunk.size()));
            if (!in.read(chunk.data(), n)) return false;
            if (!Feed(cctx.get(), out, out_buf, {chunk.data(), size_t(n), 0}, ZSTD_e_continue)) return false;
            left -= uint64_t(n);
        }
    }
    return !ec && Feed(cctx.get(), out, out_buf, {nullptr, 0, 0}, ZSTD_e_end);
}
} // namespace

bool Compress(const fs::path &src, const fs::path &dst, std::span<const std::byte> metadata) {
    std::error_code ec;
    if (!fs::is_directory(src, ec)) return false;
    const auto relative = fs::weakly_canonical(dst, ec).lexically_relative(fs::weakly_canonical(src, ec));
    if (ec || (!relative.empty() && *relative.begin() != "..")) return false;
    if (const auto parent = dst.parent_path(); !parent.empty()) fs::create_directories(parent, ec);
    if (metadata.size() > UINT32_MAX) return false;
    return !ec && bool(File::WriteAtomic(dst, [&](auto &out) {
        const uint32_t size = uint32_t(metadata.size());
        out.write(reinterpret_cast<const char *>(&MetadataMagic), sizeof(MetadataMagic));
        out.write(reinterpret_cast<const char *>(&size), sizeof(size));
        out.write(reinterpret_cast<const char *>(metadata.data()), size);
        return out && CompressToStream(src, out);
    }));
}

std::optional<std::vector<std::byte>> ReadArchiveMetadata(const fs::path &path) {
    std::ifstream in{path, std::ios::binary | std::ios::ate};
    const auto length = in.tellg();
    in.seekg(0);
    uint32_t magic{}, size{};
    if (!in.read(reinterpret_cast<char *>(&magic), sizeof(magic)) || magic != MetadataMagic ||
        !in.read(reinterpret_cast<char *>(&size), sizeof(size)) || length < std::streamoff(8ull + size)) return std::nullopt;
    std::vector<std::byte> metadata(size);
    if (!in.read(reinterpret_cast<char *>(metadata.data()), size)) return std::nullopt;
    return metadata;
}

bool Decompress(const fs::path &src, const fs::path &dst) {
    std::ifstream in{src, std::ios::binary};
    if (!in) return false;

    const std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> dctx{ZSTD_createDCtx(), ZSTD_freeDCtx};
    if (!dctx) return false;

    std::error_code ec;
    std::ofstream cur;
    uint64_t data_left = 0;
    std::vector<char> header;

    // Parse entry headers across buffer boundaries and stream entry data to the current file.
    const auto consume = [&](const char *buf, size_t n) {
        for (size_t off = 0; off < n;) {
            if (data_left > 0) {
                const size_t take = std::min<uint64_t>(data_left, n - off);
                cur.write(buf + off, std::streamsize(take));
                if (!cur) return false;

                off += take;
                if ((data_left -= take) == 0) {
                    cur.close();
                    if (!cur) return false;
                }
                continue;
            }
            header.push_back(buf[off++]);
            uint32_t path_len = 0;
            if (header.size() < sizeof path_len) continue;

            std::memcpy(&path_len, header.data(), sizeof path_len);
            if (path_len == 0 || path_len > 4096) return false;
            if (header.size() < sizeof path_len + path_len + sizeof(uint64_t)) continue;

            const fs::path entry{header.begin() + sizeof path_len, header.begin() + sizeof path_len + path_len};
            if (entry.is_absolute() || std::ranges::find(entry, "..") != entry.end()) return false;
            std::memcpy(&data_left, header.data() + sizeof path_len + path_len, sizeof data_left);
            header.clear();
            const auto path = dst / entry;
            if (const auto parent = path.parent_path(); !parent.empty()) fs::create_directories(parent, ec);
            if (ec) return false;
            cur.open(path, std::ios::binary | std::ios::trunc);
            if (!cur) return false;

            if (data_left == 0) {
                cur.close();
                if (!cur) return false;
            }
        }
        return true;
    };

    std::vector<char> in_buf(ZSTD_DStreamInSize()), out_buf(ZSTD_DStreamOutSize());
    size_t remaining = 1;
    while (in.read(in_buf.data(), std::streamsize(in_buf.size())), in.gcount() > 0) {
        ZSTD_inBuffer zin{in_buf.data(), size_t(in.gcount()), 0};
        while (zin.pos < zin.size) {
            ZSTD_outBuffer zout{out_buf.data(), out_buf.size(), 0};
            remaining = ZSTD_decompressStream(dctx.get(), &zout, &zin);
            if (ZSTD_isError(remaining) || !consume(out_buf.data(), zout.pos)) return false;
        }
    }
    // A well-formed archive ends exactly at an entry boundary.
    return in.eof() && remaining == 0 && data_left == 0 && header.empty() && !cur.is_open() && !cur.fail();
}
