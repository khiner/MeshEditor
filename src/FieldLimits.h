#pragma once

// Specializations define optional bounds for UI controls and action values.
template<auto... Ms> struct FieldLimits {};

// Use `template<> struct FieldLimits<&T::F> : Within<lo, hi> {};` for bounded fields.
template<auto Lo, auto Hi> struct Within {
    static constexpr auto Min = Lo, Max = Hi;
};
template<auto Lo> struct AtLeast {
    static constexpr auto Min = Lo;
};
template<auto Hi> struct AtMost {
    static constexpr auto Max = Hi;
};

template<auto... Ms> inline constexpr bool HasMin = requires { FieldLimits<Ms...>::Min; };
template<auto... Ms> inline constexpr bool HasMax = requires { FieldLimits<Ms...>::Max; };
template<auto... Ms> inline constexpr bool HasLimits = HasMin<Ms...> || HasMax<Ms...>;
