#pragma once

#include "PathSerialize.h"
#include "action/Action.h"
#include "action/SerializeGlm.h"

#include <cstring>
#include <istream>
#include <ostream>
#include <vector>

// Write and read actions in the `.actions` log. Each record is a uint32 byte length followed by the
// action's serialized bytes (domain index, leaf index, leaf payload), so a reader can read one record
// at a time without scanning.
namespace action {
// Append one record for `a` to `out`.
inline void SerializeAction(const Action &a, std::ostream &out) {
    static thread_local std::vector<std::byte> buffer;
    zpp::bits::out archive{buffer};
    if (zpp::bits::failure(archive(uint32_t{0}))) return; // reserve the length, filled in once known
    if (zpp::bits::failure(archive(a))) return; // a failed encode (e.g. a null owning pointer) drops the record
    const auto len = uint32_t(archive.position() - sizeof(uint32_t));
    std::memcpy(buffer.data(), &len, sizeof len);
    out.write(reinterpret_cast<const char *>(buffer.data()), std::streamsize(archive.position()));
}

// Read each action from `in` and hand it to `on_action`, one at a time so memory stays bounded regardless
// of stream length. Stops at a truncated or corrupt tail.
void StreamActions(std::istream &in, auto &&on_action) {
    std::vector<std::byte> bytes;
    uint32_t len;
    while (in.read(reinterpret_cast<char *>(&len), sizeof len)) {
        bytes.resize(len);
        if (len && !in.read(reinterpret_cast<char *>(bytes.data()), len)) return; // truncated
        Action a;
        if (zpp::bits::failure(zpp::bits::in{bytes}(a))) return; // corrupt
        on_action(std::move(a));
    }
}
} // namespace action
