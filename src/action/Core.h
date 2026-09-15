#pragma once

#include "gizmo/TransformGizmoTypes.h"
#include "gpu/DebugChannel.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"
#include "physics/PhysicsTypes.h"
#include "state/Entity.h"
#include "state/Schema.h"
#include "viewport/ViewportDisplay.h"

#include <array>
#include <concepts>
#include <limits>
#include <optional>
#include <variant>

namespace action {
// `Selected` copies the value to each selected entity.
// `SelectedDelta` offsets each selected entity by the active entity's change since the drag started.
enum class Scope : uint8_t {
    Entity,
    Active,
    Selected,
    SelectedDelta
};

// A field's value at the start of a SelectedDelta drag, so each step (and replay) computes start + delta.
struct DragFieldStart {
    state::TypeId Comp;
    uint16_t Offset, Size;
    std::array<std::byte, 16> Bytes; // fits the widest Update field (vec4)
};

// Numeric fields support SelectedDelta drags and carry scalar bounds.
template<typename F> concept ScalarField = std::floating_point<F> || (std::integral<F> && !std::same_as<F, bool>);
template<typename F> concept VectorField = requires { F::ComponentCount; } && ScalarField<typename F::value_type>;
template<typename F> inline constexpr bool DeltaField = ScalarField<F> || VectorField<F>;

// Bounds of a field that has none.
struct Unbounded {};
template<typename F> struct LimitOf {
    using Type = Unbounded;
};
template<ScalarField F> struct LimitOf<F> {
    using Type = F;
};
template<VectorField F> struct LimitOf<F> {
    using Type = typename F::value_type;
};
// The scalar type bounding a field, applied to each component of a vector.
template<typename F> using Limit = typename LimitOf<F>::Type;
template<typename L> constexpr L LowestLimit() {
    if constexpr (std::same_as<L, Unbounded>) return {};
    else return std::numeric_limits<L>::lowest();
}
template<typename L> constexpr L HighestLimit() {
    if constexpr (std::same_as<L, Unbounded>) return {};
    else return std::numeric_limits<L>::max();
}

// Writes `Value` to the field at byte `Offset` of the component on each scope target, clamped to [Min, Max].
template<typename T>
struct Update {
    Scope Scope{Scope::Entity};
    state::Entity Entity{state::Null}; // Scope::Entity only. Null targets the viewport
    state::TypeKey ComponentType;
    uint16_t Offset;
    T Value;
    Limit<T> Min{LowestLimit<Limit<T>>()}, Max{HighestLimit<Limit<T>>()};
};

// Assign authored fields together, preserving the rest of the component.
template<typename Component, typename Field, size_t N = 1>
struct PatchFields {
    state::Entity Entity;
    std::array<uint16_t, N> Offsets;
    std::array<Field, N> Values;
};

struct DestroyEntity {
    state::Entity Entity;
};

namespace detail {
template<auto> struct member_traits;
template<typename C, typename F, F C::*P>
struct member_traits<P> {
    using Class = C;
    using Field = F;
};
template<auto M> using class_of = typename member_traits<M>::Class;
template<auto M> using field_of = typename member_traits<M>::Field;

// A non-virtual data-member pointer's bit pattern is the byte offset.
template<typename P>
constexpr std::ptrdiff_t MemPtrOffset(P p) {
    static_assert(sizeof(P) == sizeof(std::ptrdiff_t));
    return std::bit_cast<std::ptrdiff_t>(p);
}

template<auto... Ms>
uint16_t FieldOffset() { return uint16_t((MemPtrOffset(Ms) + ...)); }

template<auto M, auto...> inline constexpr auto first_v = M;
template<auto... Ms> inline constexpr auto last_v = (Ms, ...);
template<auto... Ms> using first_class = class_of<first_v<Ms...>>;
template<auto... Ms> using last_field = field_of<last_v<Ms...>>;
} // namespace detail

using Core = std::variant<
    Update<bool>, Update<uint8_t>, Update<uint32_t>, Update<float>, Update<double>,
    Update<vec2>, Update<vec3>, Update<vec4>, Update<state::Entity>, Update<std::optional<uint32_t>>,
    Update<CollideMode>, Update<PhysicsCombineMode>,
    Update<TransformGizmo::Type>, Update<TransformGizmo::Mode>,
    Update<DebugChannel>, Update<AnisotropicFilterLevel>, Update<std::optional<MotionBlur>>,
    DestroyEntity>;

void Apply(state::Scene &, state::Entity viewport, const Core &);
} // namespace action
