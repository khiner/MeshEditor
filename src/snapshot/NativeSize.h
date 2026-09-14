#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>
#include <zpp_bits.h>

namespace snapshot::detail {
// Retained native payload estimate, including nested capacities. Allocator bookkeeping
// is excluded, as for the byte-store accounting; short strings are conservatively counted.
template<typename T> uint64_t NativeExtra(const T &);
struct AllocationCounter {
    static constexpr auto kind() { return zpp::bits::kind::out; }
    uint64_t operator()(const auto &...values) const { return (uint64_t{} + ... + NativeExtra(values)); }
};
template<typename T> struct NativeAllocation {
    static uint64_t Extra(const T &v) {
        if constexpr (std::is_trivially_copyable_v<T>) return 0;
        else if constexpr (requires(AllocationCounter &counter) { serialize(counter, v); }) {
            AllocationCounter counter;
            return serialize(counter, v);
        } else if constexpr (std::is_aggregate_v<T>) return zpp::bits::visit_members(v, [](const auto &...members) { return (uint64_t{} + ... + NativeExtra(members)); });
        else static_assert(std::is_trivially_copyable_v<T>, "Describe this persistent type's native allocations");
    }
};
template<typename T> uint64_t NativeExtra(const T &v) { return NativeAllocation<T>::Extra(v); }
template<typename C, typename Traits, typename A> struct NativeAllocation<std::basic_string<C, Traits, A>> {
    static uint64_t Extra(const auto &v) { return (v.capacity() + 1) * sizeof(C); }
};
template<typename T, typename A> struct NativeAllocation<std::vector<T, A>> {
    static uint64_t Extra(const auto &v) {
        uint64_t bytes = v.capacity() * sizeof(T);
        if constexpr (!std::is_trivially_copyable_v<T>)
            for (const auto &item : v) bytes += NativeExtra(item);
        return bytes;
    }
};
template<typename K, typename V, typename Compare, typename A> struct NativeAllocation<std::map<K, V, Compare, A>> {
    static uint64_t Extra(const auto &v) {
        uint64_t bytes = v.size() * (sizeof(typename std::remove_cvref_t<decltype(v)>::value_type) + 4 * sizeof(void *));
        for (const auto &[key, value] : v) bytes += NativeExtra(key) + NativeExtra(value);
        return bytes;
    }
};
template<typename K, typename V, typename H, typename E, typename A> struct NativeAllocation<std::unordered_map<K, V, H, E, A>> {
    static uint64_t Extra(const auto &v) {
        uint64_t bytes = v.bucket_count() * sizeof(void *) + v.size() * (sizeof(typename std::remove_cvref_t<decltype(v)>::value_type) + 2 * sizeof(void *));
        for (const auto &[key, value] : v) bytes += NativeExtra(key) + NativeExtra(value);
        return bytes;
    }
};
template<typename T> struct NativeAllocation<std::optional<T>> {
    static uint64_t Extra(const auto &v) { return v ? NativeExtra(*v) : 0; }
};
template<typename... T> struct NativeAllocation<std::variant<T...>> {
    static uint64_t Extra(const auto &v) {
        return std::visit([](const auto &item) { return NativeExtra(item); }, v);
    }
};
template<> struct NativeAllocation<std::filesystem::path> {
    static uint64_t Extra(const auto &v) { return NativeExtra(v.native()); }
};
} // namespace snapshot::detail
