#pragma once

#include <bit>
#include <cstdint>
#include <span>
#include <vector>

namespace store {
inline constexpr uint32_t RecordsPerPage = 32;

// A presence mask followed by length-prefixed encoded values in slot order.
// Retain scratch storage; length names the initialized prefix of the current page.
void RecordMask(std::vector<std::byte> &, uint32_t mask);
void AppendRecord(std::vector<std::byte> &, size_t &length, std::span<const std::byte>);
inline std::span<const std::byte> EncodeRecordPage(uint32_t mask, std::vector<std::byte> &out, auto &&value) {
    RecordMask(out, mask);
    size_t length = sizeof(mask);
    for (auto bits = mask; bits; bits &= bits - 1u) AppendRecord(out, length, value(uint32_t(std::countr_zero(bits))));
    return std::span{out}.first(length);
}
uint32_t RecordMask(std::span<const std::byte> &);
std::span<const std::byte> TakeRecord(std::span<const std::byte> &);
} // namespace store
