#pragma once

#include "action/Core.h"
#include "animation/AnimationData.h"
#include "gpu/PBRMaterial.h"
#include "scene/WorldTransform.h"

#include <span>

struct Material;

namespace state {
struct Registry;
} // namespace state

// Channel targets built from member pointers, and byte access to the stores they name.
namespace animation {
// A field channels can animate: floats, float vectors, and quaternions.
template<typename F>
concept KeyableField = std::floating_point<F> || std::same_as<F, quat> || (action::VectorField<F> && std::floating_point<typename F::value_type>);

template<KeyableField F> constexpr uint16_t FieldCount() {
    if constexpr (std::same_as<F, float>) return 1;
    else if constexpr (std::same_as<F, quat>) return 4;
    else return uint16_t(F::ComponentCount);
}
template<KeyableField F> constexpr ValueKind FieldKind() { return std::same_as<F, quat> ? ValueKind::Quaternion : ValueKind::Float; }

// The store a component's channels write: Transform channels write the node's pose, and PBRMaterial channels write the material entity's record in the material buffer.
template<typename C> consteval state::TypeKey StoreOf() {
    if constexpr (std::same_as<C, Transform>) return state::Key<PosedLocal>();
    else if constexpr (std::same_as<C, PBRMaterial>) return state::Key<Material>();
    else return state::Key<C>();
}

// The target of the field Ms... walks to.
template<auto... Ms>
    requires KeyableField<action::detail::last_field<Ms...>>
ChannelTarget Target() {
    using F = action::detail::last_field<Ms...>;
    return {StoreOf<action::detail::first_class<Ms...>>(), action::detail::FieldOffset<Ms...>(), FieldCount<F>(), FieldKind<F>()};
}

// Byte offset of a texture slot's TextureInfo within PBRMaterial.
uint16_t TextureSlotOffset(uint8_t texture);
// A texture transform field of a material's texture slot.
template<KeyableField F>
ChannelTarget TextureTarget(uint8_t texture, F TextureInfo::*field) {
    return {state::Key<Material>(), uint16_t(TextureSlotOffset(texture) + action::detail::MemPtrOffset(field)), FieldCount<F>(), FieldKind<F>()};
}

// The entity's morph weights, or a zero-count target for an entity without them.
ChannelTarget WeightsTarget(const state::Registry &, state::Entity);

// The translation, rotation, and scale targets of a node or bone.
std::array<ChannelTarget, 3> TransformTargets(const state::Registry &, state::Entity);

bool IsChannelStore(state::TypeKey);
std::string FieldLabel(const ChannelTarget &);

// Reads the target's current value.
// Returns false when the entity lacks the field.
// A node without a pose reads its Transform.
bool ReadField(const state::Registry &, state::Entity, const ChannelTarget &, std::span<float> out);
// Writes the target's value when it differs from the current one.
void WriteField(state::Registry &, state::Entity, const ChannelTarget &, std::span<const float>);

// Whether two values of the target are the same, allowing float rounding and quaternion sign.
bool SameValue(const ChannelTarget &, std::span<const float>, std::span<const float>);
} // namespace animation
