#include "action/Core.h"
#include "Variant.h"
#include "action/Dispatch.h"
#include "animation/AnimationData.h"
#include "armature/Armature.h"
#include "audio/AudioTypes.h"
#include "gpu/ViewportTheme.h"
#include "mesh/MeshComponents.h"
#include "physics/PhysicsTypes.h"
#include "render/MaterialComponents.h"
#include "scene/Entity.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"

namespace {
// Components whose trivially-copyable fields are written by Update<scalar>/UpdateActive<scalar>.
using UpdateableComponents = action::TypeList<
    Transform, ViewportDisplay, MaterialVariants, ArmatureAnimation, MorphWeightAnimation,
    NodeTransformAnimation, MaterialPreviewLighting, RenderedLighting, ViewportTheme,
    ViewCamera, PhysicsMaterial, CollisionFilter, ColliderMaterial, ColliderPolicy,
    PhysicsMotion, PhysicsVelocity, PhysicsJoint, TriggerNodes, ModalModelCreateInfo,
    PhysicsSimulationSettings, TransformGizmoState>;
using TagComponents = action::TypeList<SmoothShading, SubmitDirty, LightWireframeDirty, TriggerTag>;
} // namespace

namespace action {
void Apply(entt::registry &r, entt::entity viewport, const Core &action) {
    std::visit(
        overloaded{
            [&]<typename Field>(const Update<Field> &a) { ApplyUpdate<UpdateableComponents>(r, viewport, a); },
            [&]<typename Field>(const UpdateActive<Field> &a) {
                ApplyUpdate<UpdateableComponents>(r, FindActiveEntity(r), a.ComponentType, a.Offset, a.Value);
            },
            [&](const SetTag &a) { ApplyTag<TagComponents>(r, a.Entity, a.TagType, a.Present); },
            [&](const SetActiveTag &a) { ApplyTag<TagComponents>(r, FindActiveEntity(r), a.TagType, a.Present); },
            [&](const DestroyEntity &a) { r.destroy(a.Entity); },
        },
        action
    );
}
} // namespace action
