#include "Compress.h"

#include <zstd.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
constexpr int CompressionLevel{5};
constexpr size_t ChunkSize{1 << 20}; // 1 MiB file-read buffer.

// The archive is one zstd stream of entries, each: [uint32 path length][path][uint64 data length][data].
// Both sides stream through fixed buffers, so memory stays flat regardless of archive size.

// Compress `in` to `out`, writing produced bytes. `mode` is ZSTD_e_end for the final flush.
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
} // namespace

bool Compress(const fs::path &src, const fs::path &dst) {
    std::error_code ec;
    if (!fs::is_directory(src, ec)) return false;

    if (const auto parent = dst.parent_path(); !parent.empty()) fs::create_directories(parent, ec);
    std::ofstream out{dst, std::ios::binary | std::ios::trunc};
    if (!out) return false;

    std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)> cctx{ZSTD_createCCtx(), ZSTD_freeCCtx};
    if (!cctx) return false;

    ZSTD_CCtx_setParameter(cctx.get(), ZSTD_c_compressionLevel, CompressionLevel);

    std::vector<char> out_buf(ZSTD_CStreamOutSize()), chunk(ChunkSize);
    for (const auto &entry : fs::recursive_directory_iterator{src, ec}) {
        if (!entry.is_regular_file(ec)) continue;
        const auto rel = fs::relative(entry.path(), src, ec).generic_string();
        const auto data_len = uint64_t(fs::file_size(entry.path(), ec));
        if (ec) return false;

        std::ifstream in{entry.path(), std::ios::binary};
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
    return Feed(cctx.get(), out, out_buf, {nullptr, 0, 0}, ZSTD_e_end) && bool(out);
}

bool Decompress(const fs::path &src, const fs::path &dst) {
    std::ifstream in{src, std::ios::binary};
    if (!in) return false;

    std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> dctx{ZSTD_createDCtx(), ZSTD_freeDCtx};
    if (!dctx) return false;

    std::error_code ec;
    std::ofstream cur; // The file currently being written.
    uint64_t data_left = 0; // Bytes still owed to `cur`.
    std::vector<char> header; // Accumulates the next entry header between files.

    // Route decompressed bytes into the current file, or accumulate and parse the next entry header.
    const auto consume = [&](const char *buf, size_t n) {
        for (size_t off = 0; off < n;) {
            if (data_left > 0) {
                const size_t take = std::min<uint64_t>(data_left, n - off);
                cur.write(buf + off, std::streamsize(take));
                if (!cur) return false;

                off += take;
                if ((data_left -= take) == 0) cur.close();
                continue;
            }
            header.push_back(buf[off++]);
            uint32_t path_len = 0;
            if (header.size() < sizeof path_len) continue;

            std::memcpy(&path_len, header.data(), sizeof path_len);
            if (header.size() < sizeof path_len + path_len + sizeof(uint64_t)) continue;

            const std::string rel{header.data() + sizeof path_len, path_len};
            uint64_t data_len = 0;
            std::memcpy(&data_len, header.data() + sizeof path_len + path_len, sizeof data_len);
            header.clear();
            const auto path = dst / fs::path{rel};
            if (const auto parent = path.parent_path(); !parent.empty()) fs::create_directories(parent, ec);
            cur.open(path, std::ios::binary | std::ios::trunc);
            if (!cur) return false;

            data_left = data_len;
            if (data_left == 0) cur.close();
        }
        return true;
    };

    std::vector<char> in_buf(ZSTD_DStreamInSize()), out_buf(ZSTD_DStreamOutSize());
    while (in.read(in_buf.data(), std::streamsize(in_buf.size())), in.gcount() > 0) {
        ZSTD_inBuffer zin{in_buf.data(), size_t(in.gcount()), 0};
        while (zin.pos < zin.size) {
            ZSTD_outBuffer zout{out_buf.data(), out_buf.size(), 0};
            const size_t ret = ZSTD_decompressStream(dctx.get(), &zout, &zin);
            if (ZSTD_isError(ret)) return false;
            if (!consume(out_buf.data(), zout.pos)) return false;
        }
    }
    // A well-formed archive ends exactly at an entry boundary.
    return data_left == 0 && header.empty() && !cur.is_open();
}
