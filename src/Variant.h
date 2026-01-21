#pragma once

#include <variant>

template<class... Ts> struct overloaded : Ts... {
    using Ts::operator()...;
};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

// No built-in way to default-construct a variant by runtime index.
// Variants are the best, but they are the absolute worst to actually work with...
template<typename V> V CreateVariantByIndex(size_t i) {
    constexpr auto N = std::variant_size_v<V>;
    assert(i < N);
    return [&]<size_t... Is>(std::index_sequence<Is...>) -> V {
        static constexpr V table[]{V{std::in_place_index<Is>}...};
        return table[i]; // copy out
    }(std::make_index_sequence<N>{});
}
