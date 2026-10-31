#pragma once

#include "action/ActionSchema.h"
#include "gizmo/TransformGizmoTypes.h"
#include "physics/PhysicsTypes.h"
#include "viewport/ViewportDisplay.h"

enum class AttributeType : uint32_t;
enum class AttributeDomain : uint32_t;
enum class VectorMathOp : uint32_t;
enum class FloatMathOp : uint32_t;
enum class CompareOp : uint32_t;
enum class BooleanMathOp : uint32_t;
enum class BooleanOperation : uint32_t;
enum class GeometryMergeMode : uint8_t;
namespace action::mesh {
enum class DeleteMode : uint32_t;
enum class ExtrudeMode : uint8_t;
enum class DissolveMode : uint8_t;
enum class SymmetrizeAxis : uint8_t;
enum class EdgeLoopInterpolation : uint8_t;
enum class CircleFit : uint8_t;
enum class CurveElevation : uint8_t;
enum class FlattenMethod : uint8_t;
enum class BeautifyMethod : uint8_t;
} // namespace action::mesh
namespace procedure {
enum class VectorComponent : uint8_t;
} // namespace procedure
namespace primitive {
enum class FillType : uint8_t;
enum class QuadSphereMethod : uint8_t;
} // namespace primitive

namespace action {
struct OnActive {};
struct OnSelected {};
struct OnSelectedDelta {}; // Offsets each selected target by the active target's change since the drag start.
using Target = std::variant<OnActive, OnSelected, OnSelectedDelta, state::Entity>;

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

using Core = std::variant<
    Update<bool>, Update<uint8_t>, Update<uint32_t>, Update<float>, Update<double>,
    Update<vec2>, Update<uvec2>, Update<vec3>, Update<vec4>, Update<quat>, Update<state::Entity>, Update<std::optional<uint32_t>>, Update<std::optional<float>>, Update<std::optional<vec3>>,
    Update<CollideMode>, Update<PhysicsCombineMode>,
    Update<TransformGizmo::Type>, Update<TransformGizmo::Mode>,
    Update<DebugChannel>, Update<AnisotropicFilterLevel>, Update<std::optional<MotionBlur>>,
    Update<mesh::DeleteMode>, Update<GeometryMergeMode>, Update<mesh::ExtrudeMode>, Update<mesh::DissolveMode>, Update<mesh::SymmetrizeAxis>,
    Update<mesh::EdgeLoopInterpolation>, Update<mesh::CircleFit>, Update<mesh::CurveElevation>, Update<mesh::FlattenMethod>, Update<mesh::BeautifyMethod>,
    Update<AttributeType>, Update<AttributeDomain>, Update<VectorMathOp>, Update<FloatMathOp>, Update<CompareOp>, Update<BooleanMathOp>, Update<BooleanOperation>, Update<procedure::VectorComponent>, Update<primitive::FillType>, Update<primitive::QuadSphereMethod>,
    DestroyEntity>;

// Read an existing editable field as its typed Update action; invalid component/field addresses return nothing.
std::optional<Core> ReadUpdate(const state::Registry &, state::Entity, state::TypeKey, uint16_t offset);
void Apply(state::Registry &, const Core &);
// An Update records unless it targets a workspace component.
bool IsRecordable(const Core &);
} // namespace action
