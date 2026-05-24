#pragma once

#include <type_traits>
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

// Concatenate alternatives into a single std::variant. Each argument may be either a
// `std::variant<...>` (its alternatives are spliced in) or a bare type (becomes one alternative).
namespace detail {
template<typename T> struct as_variant {
    using type = std::variant<T>;
};
template<typename... Ts> struct as_variant<std::variant<Ts...>> {
    using type = std::variant<Ts...>;
};

template<typename...> struct variant_concat {
    using type = std::variant<>;
};
template<typename... Ts> struct variant_concat<std::variant<Ts...>> {
    using type = std::variant<Ts...>;
};
template<typename... Ts, typename... Us, typename... Rest>
struct variant_concat<std::variant<Ts...>, std::variant<Us...>, Rest...>
    : variant_concat<std::variant<Ts..., Us...>, Rest...> {};
} // namespace detail

template<typename... Vs>
using MergedVariantT = typename detail::variant_concat<typename detail::as_variant<Vs>::type...>::type;

// True if `T` is one of variant `V`'s alternatives. Used to route a flattened action leaf to its domain.
template<typename T, typename V> struct is_variant_member : std::false_type {};
template<typename T, typename... Ts>
struct is_variant_member<T, std::variant<Ts...>> : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};
template<typename T, typename V> inline constexpr bool is_variant_member_v = is_variant_member<T, V>::value;
