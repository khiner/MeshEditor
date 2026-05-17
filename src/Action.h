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

// Cross-cutting alts (Update<F>, Replace<X>, SetTag, DestroyEntity) — not owned by any single domain.
using CrossCuttingActions = std::variant<
    Update<bool>, Update<uint8_t>, Update<uint32_t>, Update<float>, Update<double>,
    Update<vec3>, Update<vec4>,
    Update<entt::entity>,
    Update<DebugChannel>,
    Update<CollideMode>, Update<PhysicsCombineMode>, Update<PhysicsDriveType>, Update<PhysicsDriveMode>,
    Update<vk::ClearColorValue>,
    Update<std::optional<uint32_t>>,
    SetTag, DestroyEntity,
    Replace<Camera>, Replace<MaterialDirty>, Replace<MeshMaterialAssignment>, Replace<MeshMaterialSlotSelection>,
    Replace<PhysicsMotion>, Replace<PunctualLight>, Replace<RealImpactActiveMicrophone>>;

using Action = MergedVariantT<
    selection::Actions, object::Actions, project::Actions, scene::Actions,
    physics::Actions, audio::Actions, bone::Actions, timeline::Actions,
    CrossCuttingActions>;

using FallibleAction = project::FallibleActions;
} // namespace action
