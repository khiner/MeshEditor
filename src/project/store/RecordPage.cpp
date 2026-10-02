#include "project/store/RecordPage.h"

#include <cstring>
#include <stdexcept>

namespace store {
void RecordMask(std::vector<std::byte> &out, uint32_t mask) {
    if (out.size() < sizeof(mask)) out.resize(sizeof(mask));
    std::memcpy(out.data(), &mask, sizeof(mask));
}
void AppendRecord(std::vector<std::byte> &out, size_t &length, std::span<const std::byte> value) {
    const auto first = length;
    const auto size = uint32_t(value.size());
    length += sizeof(size) + size;
    if (out.size() < length) out.resize(length);
    std::memcpy(out.data() + first, &size, sizeof(size));
    if (size) std::memcpy(out.data() + first + sizeof(size), value.data(), size);
}
uint32_t RecordMask(std::span<const std::byte> &bytes) {
    if (bytes.size() < sizeof(uint32_t)) throw std::invalid_argument("Truncated record page.");
    uint32_t word;
    std::memcpy(&word, bytes.data(), sizeof(word));
    bytes = bytes.subspan(sizeof(word));
    return word;
}
std::span<const std::byte> TakeRecord(std::span<const std::byte> &bytes) {
    const auto size = RecordMask(bytes);
    if (size > bytes.size()) throw std::invalid_argument("Truncated record value.");
    const auto value = bytes.first(size);
    bytes = bytes.subspan(size);
    return value;
}
} // namespace store
