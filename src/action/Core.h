#pragma once

#include "Field.h"
#include "gizmo/TransformGizmoTypes.h"
#include "gpu/DebugChannel.h"
#include "numeric/quat.h"
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
// Where an action applies.
struct OnActive {};
struct OnSelected {};
struct OnSelectedDelta {}; // Offsets each selected target by the active target's change since the drag start.
struct OnViewport {};
using Target = std::variant<OnActive, OnSelected, OnSelectedDelta, OnViewport, state::Entity>;

// A field's value at the start of a SelectedDelta drag, so each step (and replay) computes start + delta.
struct DragFieldStart {
    state::TypeId Comp;
    uint16_t Offset, Size;
    std::array<std::byte, 16> Bytes; // fits the widest Update field (vec4)
};

// Numeric fields support SelectedDelta drags and carry scalar bounds.
template<typename F>
concept ScalarField = std::floating_point<F> || (std::integral<F> && !std::same_as<F, bool>);
template<typename F>
concept VectorField = requires { F::ComponentCount; } && ScalarField<typename F::value_type>;
// A rotation delta composes by multiplication.
template<typename F>
concept RotationField = std::same_as<F, quat>;
template<typename F> inline constexpr bool DeltaField = ScalarField<F> || VectorField<F> || RotationField<F>;

// The scalar type bounding a numeric field, applied to each component of a vector.
template<typename F> struct LimitOf {
    using Type = F;
};
template<VectorField F> struct LimitOf<F> {
    using Type = typename F::value_type;
};
template<typename F> using Limit = typename LimitOf<F>::Type;
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

// Assigns authored fields of the active entity's component together, preserving the rest of it.
template<typename Component, typename Field, size_t N = 1>
struct PatchFields {
    std::array<uint16_t, N> Offsets;
    std::array<Field, N> Values;
};

struct DestroyEntity {
    state::Entity Entity;
};

template<typename T> inline constexpr bool IsUpdate = false;
template<typename T> inline constexpr bool IsUpdate<Update<T>> = true;
template<typename T> inline constexpr bool IsPatchFields = false;
template<typename C, typename F, size_t N> inline constexpr bool IsPatchFields<PatchFields<C, F, N>> = true;

namespace detail {
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
template<auto... Ms> using first_class = field::Owner<first_v<Ms...>>;
template<auto... Ms> using last_field = field::Type<last_v<Ms...>>;
} // namespace detail

using Core = std::variant<
    Update<bool>, Update<uint8_t>, Update<uint32_t>, Update<float>, Update<double>,
    Update<vec2>, Update<vec3>, Update<vec4>, Update<quat>, Update<state::Entity>, Update<std::optional<uint32_t>>, Update<std::optional<float>>,
    Update<CollideMode>, Update<PhysicsCombineMode>,
    Update<TransformGizmo::Type>, Update<TransformGizmo::Mode>,
    Update<DebugChannel>, Update<AnisotropicFilterLevel>, Update<std::optional<MotionBlur>>,
    DestroyEntity>;

void Apply(state::Scene &, state::Entity viewport, const Core &);
} // namespace action
