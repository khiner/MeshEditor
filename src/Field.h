#pragma once

// Compile-time field metadata derived from plain aggregates: member names and labels, enumerator names, and per-field specs.

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>

// Display and clamping properties of one field, keyed by its owner and member name.
// A radian field draws and edits in degrees.
enum class FieldUnit : uint8_t { None,
                                 Radians };
struct FieldSpec {
    double Min{-std::numeric_limits<double>::infinity()}, Max{std::numeric_limits<double>::infinity()};
    float Speed{1.f};
    uint8_t Digits{3};
    FieldUnit Unit{FieldUnit::None};

    constexpr bool HasMin() const { return Min > -std::numeric_limits<double>::infinity(); }
    constexpr bool HasMax() const { return Max < std::numeric_limits<double>::infinity(); }
    constexpr bool Bounded() const { return HasMin() || HasMax(); }
};

namespace field {
template<size_t N> struct fixed_string {
    char Chars[N]{};
    constexpr fixed_string() = default;
    consteval fixed_string(const char (&s)[N]) {
        for (size_t i = 0; i < N; ++i) Chars[i] = s[i];
    }
    consteval fixed_string(std::string_view s) {
        for (size_t i = 0; i < N - 1; ++i) Chars[i] = s[i];
    }
    constexpr std::string_view View() const { return {Chars, N - 1}; }
    constexpr const char *c_str() const { return Chars; }
};
template<size_t N> fixed_string(const char (&)[N]) -> fixed_string<N>;

namespace detail {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-var-template"
// Never defined, since only the addresses of its members are formed, as template arguments that print their names.
template<typename T> extern const T Instance;

template<typename T, size_t I> consteval auto &Member() {
    auto &[... members] = Instance<T>;
    return members...[I];
}
template<typename T, typename F> consteval size_t IndexOf(F T::*p) {
    auto &[... members] = Instance<T>;
    size_t index = sizeof...(members);
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        (..., (static_cast<const void *>(&members...[Is]) == static_cast<const void *>(&(Instance<T>.*p)) ? void(index = Is) : void()));
    }(std::make_index_sequence<sizeof...(members)>{});
    return index;
}
#pragma clang diagnostic pop

// The member name the pointer template argument prints as, such as "&Instance<T>.Name".
template<auto P> consteval std::string_view PointerName() {
    std::string_view s = __PRETTY_FUNCTION__;
    s = s.substr(0, s.rfind(']'));
    return s.substr(s.rfind('.') + 1);
}
// The enumerator name the value prints as, or empty for a value without an enumerator, which prints as a cast.
template<auto E> consteval std::string_view EnumeratorName() {
    std::string_view s = __PRETTY_FUNCTION__;
    s = s.substr(0, s.rfind(']'));
    s = s.substr(s.rfind(' ') + 1);
    if (s.find('(') != std::string_view::npos) return {};
    return s.substr(s.rfind(':') + 1);
}
// Enumerators are the named values below the probe limit, in value order.
constexpr size_t EnumProbeLimit = 64;
template<typename E> consteval size_t CountEnumerators() {
    size_t count = 0;
    [&]<size_t... Is>(std::index_sequence<Is...>) { (..., (!EnumeratorName<E(Is)>().empty() ? void(++count) : void())); }(std::make_index_sequence<EnumProbeLimit>{});
    return count;
}

constexpr bool IsUpper(char c) { return c >= 'A' && c <= 'Z'; }
constexpr bool IsLower(char c) { return c >= 'a' && c <= 'z'; }
constexpr bool IsDigit(char c) { return c >= '0' && c <= '9'; }
constexpr bool WordStart(std::string_view s, size_t i) { return i > 0 && IsUpper(s[i]) && (IsLower(s[i - 1]) || IsDigit(s[i - 1])); }
constexpr size_t SpacedSize(std::string_view s) {
    size_t n = s.size();
    for (size_t i = 0; i < s.size(); ++i) n += WordStart(s, i);
    return n;
}
constexpr void SpaceWords(std::string_view s, char *out) {
    for (size_t i = 0; i < s.size(); ++i) {
        if (WordStart(s, i)) *out++ = ' ';
        *out++ = s[i];
    }
}
// "OuterConeAngle" reads "Outer Cone Angle".
template<fixed_string S> consteval auto Spaced() {
    fixed_string<SpacedSize(S.View()) + 1> out{};
    SpaceWords(S.View(), out.Chars);
    return out;
}

template<auto> struct MemberTraits;
template<typename C, typename F, F C::*P> struct MemberTraits<P> {
    using Owner = C;
    using Field = F;
};
} // namespace detail

template<auto P> using Owner = typename detail::MemberTraits<P>::Owner;
template<auto P> using Type = typename detail::MemberTraits<P>::Field;

template<typename T, size_t I> inline constexpr std::string_view Name = detail::PointerName<&detail::Member<T, I>()>();
template<typename T, size_t I> inline constexpr auto NameString = fixed_string<Name<T, I>.size() + 1>{Name<T, I>};
template<typename T, size_t I> inline constexpr auto Label = detail::Spaced<NameString<T, I>>();
template<auto P> inline constexpr size_t IndexOf = detail::IndexOf<Owner<P>, Type<P>>(P);
template<auto P> inline constexpr auto LabelOf = Label<Owner<P>, IndexOf<P>>;

template<typename E> inline constexpr size_t EnumCount = detail::CountEnumerators<E>();
template<typename E, size_t I> inline constexpr auto EnumeratorLabel = detail::Spaced<fixed_string<detail::EnumeratorName<E(I)>().size() + 1>{detail::EnumeratorName<E(I)>()}>();
template<typename E> inline constexpr auto EnumValues = []<size_t... Is>(std::index_sequence<Is...>) {
    std::array<E, EnumCount<E>> out{};
    size_t j = 0;
    (..., (!detail::EnumeratorName<E(Is)>().empty() ? void(out[j++] = E(Is)) : void()));
    return out;
}(std::make_index_sequence<detail::EnumProbeLimit>{});
template<typename E> inline constexpr auto EnumLabels = []<size_t... Is>(std::index_sequence<Is...>) {
    return std::array<const char *, EnumCount<E>>{EnumeratorLabel<E, size_t(EnumValues<E>[Is])>.c_str()...};
}(std::make_index_sequence<EnumCount<E>>{});

template<typename T> inline constexpr bool IsArray = false;
template<typename T, size_t N> inline constexpr bool IsArray<std::array<T, N>> = true;
template<typename T>
concept Walkable = std::is_class_v<T> && std::is_aggregate_v<T> && !IsArray<T>;

// Visits each member with its declaration index, as `f(member, std::integral_constant<size_t, I>{})`.
template<typename T, typename F> constexpr void ForEach(T &value, F &&f) {
    auto &[... members] = value;
    [&]<size_t... Is>(std::index_sequence<Is...>) { (..., f(members... [Is], std::integral_constant<size_t, Is> {})); }(std::make_index_sequence<sizeof...(members)>{});
}
} // namespace field

// Declare `template<> inline constexpr FieldSpec Spec<T, "Member">{.Min = lo, .Max = hi};` beside the owning type.
template<typename T, field::fixed_string Name> inline constexpr FieldSpec Spec{};

namespace field {
template<auto P> inline constexpr const FieldSpec &SpecOf = Spec<Owner<P>, NameString<Owner<P>, IndexOf<P>>>;

// Bounds come from the innermost member that declares any.
template<auto... Ms> consteval FieldSpec ChainSpec() {
    constexpr std::array specs{SpecOf<Ms>...};
    FieldSpec spec = specs.back();
    for (size_t i = specs.size() - 1; i-- > 0 && !spec.Bounded();) {
        spec.Min = specs[i].Min;
        spec.Max = specs[i].Max;
    }
    return spec;
}
} // namespace field
