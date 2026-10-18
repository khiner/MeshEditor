#pragma once

#include "state/Schema.h"

#include <array>
#include <optional>
#include <string_view>
#include <type_traits>

// A radian field displays in degrees.
enum class FieldUnit : uint8_t { None,
                                 Radians,
                                 Degrees,
                                 Meters,
                                 MetersPerSecond,
                                 RadiansPerSecond,
                                 Kilograms,
                                 KilogramsPerCubicMeter,
                                 Pascals,
                                 PerSecond,
                                 Seconds,
                                 Hertz,
                                 Nanometers,
                                 Stops,
                                 Points,
                                 Pixels,
                                 Cells,
                                 Frames,
                                 Factor };
// The symbol shown after a value, in enumerator order.
inline constexpr std::string_view UnitSymbols[]{"", "°", "°", " m", " m/s", " rad/s", " kg", " kg/m³", " Pa", " 1/s", " s", " Hz", " nm", " EV", " pt", " px", " cells", " frames", "x"};
constexpr std::string_view UnitSymbol(FieldUnit unit) { return UnitSymbols[size_t(unit)]; }
// Decimals a unit's values show by default.
constexpr uint8_t UnitDigits(FieldUnit unit) {
    switch (unit) {
        case FieldUnit::Radians:
        case FieldUnit::Degrees:
        case FieldUnit::Stops:
        case FieldUnit::Points:
        case FieldUnit::Pixels: return 1;
        case FieldUnit::Hertz:
        case FieldUnit::Nanometers:
        case FieldUnit::KilogramsPerCubicMeter: return 0;
        default: return 3;
    }
}
// Auto picks a checkbox for a bool, a combo for an enum, the rotation widget for a quaternion, a slider for a bounded number, and a drag otherwise.
// Checkbox toggles an integer between 0 and 1.
enum class FieldWidget : uint8_t { Auto,
                                   Drag,
                                   Slider,
                                   Color,
                                   Checkbox };
enum class FieldScale : uint8_t { Linear,
                                  Logarithmic };
struct FieldSpec {
    double Min{-std::numeric_limits<double>::infinity()}, Max{std::numeric_limits<double>::infinity()};
    double Initial{std::numeric_limits<double>::quiet_NaN()}; // The value an enabled optional field starts with, or NaN for the clamped type default
    float Speed{1.f};
    std::optional<uint8_t> Digits{}; // Decimals shown for a floating-point value, or the unit's default
    FieldUnit Unit{FieldUnit::None};
    FieldWidget Widget{FieldWidget::Auto};
    FieldScale Scale{FieldScale::Linear};
    const char *Label{}; // Overrides the spaced field name
    const char *Format{}; // Overrides the number conversion built from Digits, ahead of the unit symbol
    const char *Help{}; // A description of the field

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
template<typename T> consteval size_t Count() {
    auto &[... members] = Instance<T>;
    return sizeof...(members);
}
#pragma clang diagnostic pop

template<auto P> consteval std::string_view PointerName() {
    std::string_view s = __PRETTY_FUNCTION__;
    s = s.substr(0, s.rfind(']'));
    return s.substr(s.rfind('.') + 1);
}
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
constexpr char *Append(char *out, std::string_view s) {
    for (const char c : s) *out++ = c;
    return out;
}
} // namespace detail

// Member I of T, usable in a field chain in place of a member pointer.
template<typename T, size_t I> struct At {
    using Owner = T;
    using Field = std::remove_cvref_t<decltype(detail::Member<T, I>())>;
    static constexpr size_t Index = I;
};
template<typename T> inline constexpr bool IsAt = false;
template<typename T, size_t I> inline constexpr bool IsAt<At<T, I>> = true;

namespace detail {
template<auto> struct MemberTraits;
template<typename C, typename F, F C::*P> struct MemberTraits<P> {
    using Owner = C;
    using Field = F;
};
template<auto P>
    requires IsAt<std::remove_cv_t<decltype(P)>>
struct MemberTraits<P> {
    using Owner = typename std::remove_cv_t<decltype(P)>::Owner;
    using Field = typename std::remove_cv_t<decltype(P)>::Field;
};
template<auto M, auto...> inline constexpr auto First = M;
template<auto... Ms> inline constexpr auto Last = (Ms, ...);
} // namespace detail

template<auto P> using Owner = typename detail::MemberTraits<P>::Owner;
template<auto P> using Type = typename detail::MemberTraits<P>::Field;

template<typename T, size_t I> inline constexpr std::string_view Name = detail::PointerName<&detail::Member<T, I>()>();
template<typename T, size_t I> inline constexpr auto NameString = fixed_string<Name<T, I>.size() + 1>{Name<T, I>};
template<typename T, size_t I> inline constexpr auto Label = detail::Spaced<NameString<T, I>>();
template<auto P> inline constexpr size_t IndexOf = [] {
    if constexpr (IsAt<std::remove_cv_t<decltype(P)>>) return std::remove_cv_t<decltype(P)>::Index;
    else return detail::IndexOf<Owner<P>, Type<P>>(P);
}();
template<auto P> inline constexpr auto LabelOf = Label<Owner<P>, IndexOf<P>>;

template<typename T> inline constexpr auto TypeLabel = detail::Spaced<fixed_string<state::LeafName<T>().size() + 1>{state::LeafName<T>()}>();

namespace detail {
template<typename T> consteval std::string_view TypeNameView() {
    if constexpr (std::same_as<T, bool>) return "bool";
    else if constexpr (std::same_as<T, float>) return "float";
    else if constexpr (std::same_as<T, double>) return "double";
    else if constexpr (std::same_as<T, int8_t>) return "int8";
    else if constexpr (std::same_as<T, uint8_t>) return "uint8";
    else if constexpr (std::same_as<T, int16_t>) return "int16";
    else if constexpr (std::same_as<T, uint16_t>) return "uint16";
    else if constexpr (std::same_as<T, int32_t>) return "int32";
    else if constexpr (std::same_as<T, uint32_t>) return "uint32";
    else if constexpr (std::same_as<T, int64_t>) return "int64";
    else if constexpr (std::same_as<T, uint64_t>) return "uint64";
    else return state::LeafName<T>();
}
} // namespace detail
template<typename T> inline constexpr auto TypeName = fixed_string<detail::TypeNameView<T>().size() + 1>{detail::TypeNameView<T>()};

template<auto M> auto &Get(auto &obj) {
    if constexpr (IsAt<std::remove_cv_t<decltype(M)>>) {
        auto &[... members] = obj;
        return members...[std::remove_cv_t<decltype(M)>::Index];
    } else {
        return obj.*M;
    }
}

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
constexpr void ForEach(auto &value, auto &&f) {
    auto &[... members] = value;
    [&]<size_t... Is>(std::index_sequence<Is...>) { (..., f(members... [Is], std::integral_constant<size_t, Is> {})); }(std::make_index_sequence<sizeof...(members)>{});
}
} // namespace field

namespace field {
void DuplicateEntry(); // Never defined, so a second entry for one field fails to compile.
// A value per field of T, indexed by field, with fields left out holding the default value.
template<typename T, typename V> struct Table {
    struct Entry {
        size_t Index;
        V Value;
        consteval Entry(auto T::*member, V value) : Index{detail::IndexOf(member)}, Value{value} {}
    };
    std::array<V, detail::Count<T>()> ByIndex{};
    constexpr Table() = default;
    consteval Table(std::initializer_list<Entry> entries) {
        std::array<bool, detail::Count<T>()> declared{};
        for (const auto &entry : entries) {
            if (declared[entry.Index]) DuplicateEntry();
            declared[entry.Index] = true;
            ByIndex[entry.Index] = entry.Value;
        }
    }
};
template<typename T> using SpecTable = Table<T, FieldSpec>;
} // namespace field

// Declare `template<> inline constexpr field::SpecTable<T> Specs<T>{{&T::Member, {.Min = lo, .Max = hi}}, ...};` beside the owning type.
template<typename T> inline constexpr field::SpecTable<T> Specs{};
template<typename T, size_t I> inline constexpr const FieldSpec &Spec = Specs<T>.ByIndex[I];

namespace field {
template<auto P> inline constexpr const FieldSpec &SpecOf = Spec<Owner<P>, IndexOf<P>>;

// Bounds and Widget come from the innermost chain member that sets them.
template<auto... Ms> consteval FieldSpec ChainSpec() {
    constexpr std::array specs{SpecOf<Ms>...};
    FieldSpec spec = specs.back();
    for (size_t i = specs.size() - 1; i-- > 0 && !spec.Bounded();) {
        spec.Min = specs[i].Min;
        spec.Max = specs[i].Max;
    }
    for (size_t i = specs.size() - 1; i-- > 0 && spec.Widget == FieldWidget::Auto;) spec.Widget = specs[i].Widget;
    return spec;
}
template<auto... Ms> inline constexpr FieldSpec ChainSpecOf = ChainSpec<Ms...>();

template<auto P> consteval std::string_view MemberLabel() { return SpecOf<P>.Label ? std::string_view{SpecOf<P>.Label} : LabelOf<P>.View(); }

namespace detail {
template<auto... Ms> consteval auto MakeChainPath() {
    constexpr std::array labels{MemberLabel<Ms>()...};
    constexpr std::string_view component = TypeLabel<Owner<First<Ms...>>>.View();
    constexpr size_t size = [&] {
        size_t n = component.size();
        for (size_t i = 0; i + 1 < labels.size(); ++i) n += 3 + labels[i].size();
        return n;
    }();
    fixed_string<size + 1> out{};
    char *at = Append(out.Chars, component);
    for (size_t i = 0; i + 1 < labels.size(); ++i) at = Append(Append(at, " / "), labels[i]);
    return out;
}
} // namespace detail

// The leaf's label: its spec Label or spaced name.
template<auto... Ms> inline constexpr auto ChainLabel = fixed_string<MemberLabel<detail::Last<Ms...>>().size() + 1>{MemberLabel<detail::Last<Ms...>>()};
// The component name and each enclosing member's label, joined by " / ".
template<auto... Ms> inline constexpr auto ChainPath = detail::MakeChainPath<Ms...>();
} // namespace field
