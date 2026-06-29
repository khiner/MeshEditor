#pragma once

#include "action/Action.h"

#include <zpp_bits.h>

#include <array>
#include <istream>
#include <ostream>
#include <span>
#include <vector>

// Write and read actions in the `.actions` log. Each record is one action: a uint32 variant index followed by
// the action's serialized bytes. For byte-serializable alternatives the bytes are raw and fixed-length
// (sizeof, or 0 for an empty struct); other alternatives prefix their bytes with a uint32 length, so a
// reader can read one record at a time without scanning.
namespace action {
namespace detail {
// Serialized byte count per alternative when fixed-length, or nullopt when length-prefixed.
inline constexpr auto FixedSizes = []<size_t... Is>(std::index_sequence<Is...>) {
    return std::array<std::optional<size_t>, sizeof...(Is)>{
        [] -> std::optional<size_t> {
            using Alt = std::variant_alternative_t<Is, Action>;
            if constexpr (zpp::bits::concepts::byte_serializable<Alt>) return std::is_empty_v<Alt> ? 0 : sizeof(Alt);
            else return std::nullopt;
        }()...
    };
}(std::make_index_sequence<std::variant_size_v<Action>>{});

// Construct alternative `index` of Action and decode it from `bytes`. False on bad index or decode error.
inline bool DecodeAlternative(Action &a, uint32_t index, std::span<const std::byte> bytes) {
    bool ok = false;
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((Is == index ? (void)(a.emplace<Is>(), ok = !zpp::bits::failure(zpp::bits::in{bytes}(std::get<Is>(a)))) : void()), ...);
    }(std::make_index_sequence<std::variant_size_v<Action>>{});
    return ok;
}
} // namespace detail

// Append one record for `a` to `out`.
inline void SerializeAction(const Action &a, std::ostream &out) {
    static thread_local std::vector<std::byte> buffer;
    zpp::bits::out archive{buffer};
    if (zpp::bits::failure(archive(uint32_t(a.index())))) return;
    const bool ok = std::visit(
        [&](const auto &alt) {
            using Alt = std::decay_t<decltype(alt)>;
            if constexpr (zpp::bits::concepts::byte_serializable<Alt>) {
                return !zpp::bits::failure(archive(alt));
            } else {
                const auto len_pos = archive.position();
                if (zpp::bits::failure(archive(uint32_t{0}))) return false; // reserve the length; filled in once known
                const auto start = archive.position();
                if (zpp::bits::failure(archive(alt))) return false; // e.g. a null owning pointer; drop the record
                const auto len = uint32_t(archive.position() - start);
                std::memcpy(buffer.data() + len_pos, &len, sizeof len);
                return true;
            }
        },
        a
    );
    if (!ok) return;
    out.write(reinterpret_cast<const char *>(buffer.data()), std::streamsize(archive.position()));
}

// Read each action from `in` and hand it to `on_action`, one at a time so memory stays bounded regardless
// of stream length. Stops at a truncated or corrupt tail.
void StreamActions(std::istream &in, auto &&on_action) {
    std::vector<std::byte> bytes;
    uint32_t index;
    while (in.read(reinterpret_cast<char *>(&index), sizeof index)) {
        if (index >= detail::FixedSizes.size()) return; // corrupt index
        uint32_t len;
        if (const auto fixed = detail::FixedSizes[index]) len = uint32_t(*fixed);
        else if (!in.read(reinterpret_cast<char *>(&len), sizeof len)) return; // truncated
        bytes.resize(len);
        if (len && !in.read(reinterpret_cast<char *>(bytes.data()), len)) return; // truncated
        Action a;
        if (!detail::DecodeAlternative(a, index, std::span{bytes})) return; // corrupt
        on_action(std::move(a));
    }
}
} // namespace action
