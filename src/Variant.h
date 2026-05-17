#pragma once

#include <variant>

template<class... Ts> struct overloaded : Ts... {
    using Ts::operator()...;
};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

// Default-construct a variant by runtime index.
template<typename V> V CreateVariantByIndex(size_t i) {
    constexpr auto N = std::variant_size_v<V>;
    return [&]<size_t... Is>(std::index_sequence<Is...>) -> V {
        static constexpr V table[]{V{std::in_place_index<Is>}...};
        return table[i]; // copy out
    }(std::make_index_sequence<N>{});
}

// Concatenate the alternatives of multiple std::variant types into one.
template<typename...> struct MergedVariant {
    using type = std::variant<>;
};
template<typename... Ts> struct MergedVariant<std::variant<Ts...>> {
    using type = std::variant<Ts...>;
};
template<typename... Ts, typename... Us, typename... Rest>
struct MergedVariant<std::variant<Ts...>, std::variant<Us...>, Rest...>
    : MergedVariant<std::variant<Ts..., Us...>, Rest...> {};

template<typename... Vs>
using MergedVariantT = MergedVariant<Vs...>::type;
