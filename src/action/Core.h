#pragma once

#include "action/ActionSchema.h"
#include "action/Emit.h"
#include "viewport/ViewportDisplay.h"

#include <variant>

namespace state {
struct DirtySet;
}

namespace action {
struct OnActive {};
struct OnSelected {};
struct OnSelectedDelta {}; // Offsets each selected target by the active target's change since the drag start.
using Target = std::variant<OnActive, OnSelected, OnSelectedDelta, state::Entity>;

// A field's value at the start of a SelectedDelta drag, from which each step and its replay compute start + delta.
struct DragFieldStart {
    state::TypeId Comp;
    uint16_t Offset, Size;
    std::array<std::byte, 16> Bytes; // Fits the widest Update field, a vec4
};

template<typename F>
concept ScalarField = std::floating_point<F> || (std::integral<F> && !std::same_as<F, bool>);
template<typename F>
concept VectorField = requires { F::ComponentCount; } && ScalarField<typename F::value_type>;
template<typename F>
concept RotationField = std::same_as<F, quat>;
template<typename F> inline constexpr bool DeltaField = ScalarField<F> || VectorField<F> || RotationField<F>;

template<typename F> struct LimitOf {
    using Type = F;
};
template<VectorField F> struct LimitOf<F> {
    using Type = F::value_type;
};
template<typename F> using Limit = LimitOf<F>::Type;
template<typename F> inline constexpr int Components = 1;
template<VectorField F> inline constexpr int Components<F> = int(F::ComponentCount);

// A spec's bounds as the field's limit type, with an open endpoint at the type's extreme.
template<typename F> constexpr Limit<F> LowerBound(const FieldSpec &spec) { return spec.HasMin() ? Limit<F>(spec.Min) : std::numeric_limits<Limit<F>>::lowest(); }
template<typename F> constexpr Limit<F> UpperBound(const FieldSpec &spec) { return spec.HasMax() ? Limit<F>(spec.Max) : std::numeric_limits<Limit<F>>::max(); }

// Writes `Value` to the field at byte `Offset` of the component on each target, clamped to the field's spec.
template<typename T>
struct Update {
    Target Target{OnActive{}};
    state::TypeKey ComponentType;
    uint16_t Offset;
    T Value;
};

// Assigns the field at byte `Offset` of the component, creating the component from domain defaults when absent.
template<typename Component, typename Field>
struct PatchFields {
    uint16_t Offset;
    Field Value;
    state::Entity Entity{state::Null};
};

struct DestroyEntity {
    state::Entity Entity;
};

template<typename T> inline constexpr bool IsUpdate = false;
template<typename T> inline constexpr bool IsUpdate<Update<T>> = true;
template<typename T> inline constexpr bool IsPatchFields = false;
template<typename C, typename F> inline constexpr bool IsPatchFields<PatchFields<C, F>> = true;

namespace detail {
// A non-virtual data-member pointer's bit pattern is the byte offset.
constexpr std::ptrdiff_t MemPtrOffset(auto p) { return std::bit_cast<std::ptrdiff_t>(p); }
// The byte offset of member I, measured once on a default-constructed T.
template<typename T, size_t I>
std::ptrdiff_t MemPtrOffset(field::At<T, I>) {
    static const auto offset = [] {
        const T value{};
        auto &[... members] = value;
        return reinterpret_cast<const std::byte *>(&members...[I]) - reinterpret_cast<const std::byte *>(&value);
    }();
    return offset;
}

template<auto... Ms>
uint16_t FieldOffset() { return uint16_t((MemPtrOffset(Ms) + ...)); }

template<auto M, auto...> inline constexpr auto first_v = M;
template<auto... Ms> inline constexpr auto last_v = (Ms, ...);
template<auto... Ms> using first_class = field::Owner<first_v<Ms...>>;
template<auto... Ms> using last_field = field::Type<last_v<Ms...>>;
} // namespace detail

// An enumerated field updates as the unsigned integer of its size, and its component and offset identify the enum.
template<typename F>
concept EnumeratedField = std::is_enum_v<F> && field::Enumerated<F>;
template<typename F> struct UpdateValueOf {
    using Type = F;
};
template<EnumeratedField F> struct UpdateValueOf<F> {
    static_assert(sizeof(F) == 1 || sizeof(F) == 4, "An enumerated field in an Update must be one or four bytes");
    using Type = std::conditional_t<sizeof(F) == 1, uint8_t, uint32_t>;
};
template<typename F> using UpdateValue = UpdateValueOf<F>::Type;
template<typename F> constexpr UpdateValue<F> ToUpdateValue(F v) { return UpdateValue<F>(v); }

using Core = std::variant<
    Update<bool>, Update<uint8_t>, Update<uint32_t>, Update<float>, Update<double>,
    Update<vec2>, Update<uvec2>, Update<vec3>, Update<vec4>, Update<quat>, Update<state::Entity>,
    Update<std::optional<uint32_t>>, Update<std::optional<float>>, Update<std::optional<vec3>>, Update<std::optional<MotionBlur>>,
    DestroyEntity>;
void Append(Destination, Core);

// Reads an editable field as its typed Update action, or nothing for an invalid component or field address.
std::optional<Core> ReadUpdate(const state::Registry &, state::Entity, state::TypeKey, uint16_t offset);
// Adds to the set each entity that loses a component ReadUpdate reads.
void TrackFieldRemovals(state::DirtySet &);
// The entity an Update of the component addresses for the active target, or null when there is none.
state::Entity ActiveUpdateTarget(const state::Registry &, state::TypeKey component);
void Apply(state::Registry &, const Core &);
// An Update records unless it targets a workspace component.
bool IsRecordable(const Core &);
} // namespace action
