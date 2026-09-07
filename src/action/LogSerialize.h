#pragma once

#include "PathSerialize.h"
#include "action/Action.h"
#include "numeric/Serialize.h"

#include <cstring>
#include <istream>
#include <limits>
#include <ostream>
#include <vector>

// Each record contains a uint32 byte length followed by its domain index, leaf index, and payload.
namespace action {
inline void SerializeAction(const Action &a, std::ostream &out) {
    static thread_local std::vector<std::byte> buffer;
    zpp::bits::out archive{buffer};
    if (zpp::bits::failure(archive(uint32_t{0}))) return;
    if (zpp::bits::failure(archive(a))) return;
    const auto len = uint32_t(archive.position() - sizeof(uint32_t));
    std::memcpy(buffer.data(), &len, sizeof len);
    out.write(reinterpret_cast<const char *>(buffer.data()), std::streamsize(archive.position()));
}

// Streams actions to `on_action` with bounded memory and stops at a truncated or corrupt record.
void StreamActions(std::istream &in, auto &&on_action, uint64_t count = std::numeric_limits<uint64_t>::max()) {
    std::vector<std::byte> bytes;
    uint32_t len;
    for (uint64_t i = 0; i < count && in.read(reinterpret_cast<char *>(&len), sizeof len); ++i) {
        bytes.resize(len);
        if (len && !in.read(reinterpret_cast<char *>(bytes.data()), len)) return;
        Action a;
        if (zpp::bits::failure(zpp::bits::in{bytes}(a))) return;
        on_action(std::move(a));
    }
}
} // namespace action
