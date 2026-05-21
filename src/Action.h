#pragma once

#include "Variant.h"
#include "action/Audio.h"
#include "action/Bone.h"
#include "action/Core.h"
#include "action/Object.h"
#include "action/Physics.h"
#include "action/Project.h"
#include "action/Scene.h"
#include "action/Selection.h"
#include "action/Timeline.h"
#include "audio/RealImpactComponents.h"
#include "gpu/PunctualLight.h"
#include "scene_impl/SceneComponents.h"

namespace action {

using Action = MergedVariantT<
    Core, selection::Actions, object::Actions, project::Actions, scene::Actions,
    physics::Actions, audio::Actions, bone::Actions, timeline::Actions,
    Update<DebugChannel>, Update<CollideMode>,
    Update<PhysicsCombineMode>, Update<PhysicsDriveType>, Update<PhysicsDriveMode>, Update<vk::ClearColorValue>,
    Update<std::optional<uint32_t>>,
    Update<TransformGizmo::Type>, Update<TransformGizmo::Mode>,
    Replace<Camera>, Replace<MaterialDirty>, Replace<MeshMaterialAssignment>, Replace<MeshMaterialSlotSelection>,
    Replace<PhysicsMotion>, Replace<PunctualLight>, Replace<RealImpactActiveMicrophone>,
    ReplaceActive<Camera>, ReplaceActive<PhysicsMotion>, ReplaceActive<PunctualLight>>;

using FallibleAction = project::FallibleActions;

// Promote a sub-domain Action variant into the top-level Action.
template<typename... Ts>
void Assign(std::optional<Action> &out, std::variant<Ts...> v) {
    std::visit([&](auto &&x) { out = std::forward<decltype(x)>(x); }, std::move(v));
}
} // namespace action
