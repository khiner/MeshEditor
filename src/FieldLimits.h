#pragma once

// A field's value range. Unbounded by default, specialize with an optional Min and/or Max.
// Read by the UI (widget bounds) and by action apply (clamp).
template<auto... Ms> struct FieldLimits {};

template<auto... Ms> inline constexpr bool HasMin = requires { FieldLimits<Ms...>::Min; };
template<auto... Ms> inline constexpr bool HasMax = requires { FieldLimits<Ms...>::Max; };
template<auto... Ms> inline constexpr bool HasLimits = HasMin<Ms...> || HasMax<Ms...>;
