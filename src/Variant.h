#pragma once

#include <variant>

template<typename... Ts> struct overloaded : Ts... {
    using Ts::operator()...;
};
template<typename... Ts> overloaded(Ts...) -> overloaded<Ts...>;

template<typename V> V CreateVariantByIndex(size_t i) {
    constexpr auto N = std::variant_size_v<V>;
    return [&]<size_t... Is>(std::index_sequence<Is...>) -> V {
        static constexpr V table[]{V{std::in_place_index<Is>}...};
        return table[i];
    }(std::make_index_sequence<N>{});
}

template<typename T, typename V> struct is_variant_member : std::false_type {};
template<typename T, typename... Ts>
struct is_variant_member<T, std::variant<Ts...>> : std::bool_constant<(std::same_as<T, Ts> || ...)> {};
template<typename T, typename V> inline constexpr bool is_variant_member_v = is_variant_member<T, V>::value;
