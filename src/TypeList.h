#pragma once

#include <concepts>

template<typename... Ts> struct TypeList {};

template<typename T, typename List> inline constexpr bool InTypeList = false;
template<typename T, typename... Ts> inline constexpr bool InTypeList<T, TypeList<Ts...>> = (std::same_as<T, Ts> || ...);

template<typename... As, typename... Bs> constexpr TypeList<As..., Bs...> Concat(TypeList<As...>, TypeList<Bs...>) { return {}; }
