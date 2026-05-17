#pragma once

#include "Variant.h"
#include "action/Audio.h"
#include "action/Bone.h"
#include "action/Destroy.h"
#include "action/Object.h"
#include "action/Physics.h"
#include "action/Project.h"
#include "action/Replace.h"
#include "action/Scene.h"
#include "action/Selection.h"
#include "action/Tag.h"
#include "action/Timeline.h"
#include "action/Update.h"
#include "audio/RealImpactComponents.h"
#include "gpu/PunctualLight.h"
#include "numeric/vec4.h"
#include "scene_impl/SceneComponents.h"

namespace action {

// `*Active` actions use FindActiveEntity instead of storing the active entity.
using CrossCuttingActions = std::variant<
    Update<bool>, Update<uint8_t>, Update<uint32_t>, Update<float>, Update<double>,
    Update<vec3>, Update<vec4>,
    Update<entt::entity>,
    Update<DebugChannel>,
    Update<CollideMode>, Update<PhysicsCombineMode>, Update<PhysicsDriveType>, Update<PhysicsDriveMode>,
    Update<vk::ClearColorValue>,
    Update<std::optional<uint32_t>>,
    UpdateActive<bool>, UpdateActive<uint32_t>, UpdateActive<float>, UpdateActive<double>, UpdateActive<vec3>, UpdateActive<entt::entity>,
    SetTag, SetActiveTag, DestroyEntity,
    Replace<Camera>, Replace<MaterialDirty>, Replace<MeshMaterialAssignment>, Replace<MeshMaterialSlotSelection>,
    Replace<PhysicsMotion>, Replace<PunctualLight>, Replace<RealImpactActiveMicrophone>,
    ReplaceActive<Camera>, ReplaceActive<PhysicsMotion>, ReplaceActive<PunctualLight>>;

using Action = MergedVariantT<
    selection::Actions, object::Actions, project::Actions, scene::Actions,
    physics::Actions, audio::Actions, bone::Actions, timeline::Actions,
    CrossCuttingActions>;

using FallibleAction = project::FallibleActions;

// Promote a sub-domain Action variant into the top-level Action.
template<typename... Ts>
void Assign(std::optional<Action> &out, std::variant<Ts...> v) {
    std::visit([&](auto &&x) { out = std::forward<decltype(x)>(x); }, std::move(v));
}
} // namespace action
