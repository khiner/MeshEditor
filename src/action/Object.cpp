#include "action/Object.h"
#include "Timer.h"
#include "action/Dispatch.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "object/ObjectOps.h"
#include "render/GpuBufferOps.h"
#include "render/Instance.h"
#include "render/LightComponents.h"
#include "render/MeshBuffers.h"
#include "scene/SceneGraphOps.h"
#include "scene/WorldTransform.h"
#include "selection/Selection.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"

using std::ranges::to;

namespace {
entt::entity DuplicateOne(entt::registry &r, entt::entity e, bool &was_mesh_duplicate) {
    auto &meshes = r.ctx().get<MeshStore>();
    const ObjectCreateInfo create_info{
        .Name = std::format("{}_copy", GetName(r, e)),
        // Duplicate is created at root, so its local must match source's world.
        .Transform = Transform{r.get<const WorldTransform>(e)},
        .Select = r.all_of<Selected>(e) ? MeshInstanceCreateInfo::SelectBehavior::Additive : MeshInstanceCreateInfo::SelectBehavior::None,
    };

    if (!r.all_of<Instance>(e)) {
        if (const auto object_type = r.all_of<ObjectKind>(e) ? r.get<const ObjectKind>(e).Value : ObjectType::Empty; object_type == ObjectType::Armature) {
            const auto data_entity = r.create();
            r.emplace<Armature>(data_entity);
            const auto copy_entity = r.create();
            r.emplace<ObjectKind>(copy_entity, ObjectType::Armature);
            r.emplace<ArmatureObject>(copy_entity, data_entity);
            r.emplace<Transform>(copy_entity, create_info.Transform);
            r.emplace<WorldTransform>(copy_entity, create_info.Transform);
            r.emplace<Name>(copy_entity, ::CreateName(r, create_info.Name.empty() ? "Armature" : create_info.Name));
            ::ApplySelectBehavior(r, copy_entity, create_info.Select);
            ::CreateBoneInstances(r, meshes, copy_entity, data_entity);
            if (const auto *src_armature = r.try_get<ArmatureObject>(e)) {
                auto &dst = r.get<Armature>(data_entity);
                dst = r.get<const Armature>(src_armature->Entity);
            }
            return copy_entity;
        }
        return ::AddEmpty(r, meshes, create_info);
    }

    // Bone sub-entities (head/tail joints, bone instances) are not independently duplicable.
    if (r.all_of<BoneSubPartOf>(e)) return entt::null;

    // Object extras (Camera, Empty, Light) have Instance but create their own wireframe mesh.
    if (r.all_of<ObjectExtrasTag>(r.get<Instance>(e).Entity)) {
        if (const auto *src_cd = r.try_get<Camera>(e)) return ::AddCamera(r, meshes, create_info, *src_cd);
        if (r.all_of<LightIndex>(e)) return ::AddLight(r, meshes, create_info, GetLight(r, r.get<const LightIndex>(e).Value));
        return ::AddEmpty(r, meshes, create_info);
    }

    const auto mesh_entity = r.get<Instance>(e).Entity;
    const auto e_new = ::AddMesh(
        r, meshes,
        meshes.CloneMesh(r.get<const Mesh>(mesh_entity)),
        MeshInstanceCreateInfo{.Name = create_info.Name, .Transform = create_info.Transform, .Select = create_info.Select, .Visible = r.all_of<RenderInstance>(e)}
    );
    if (auto prim_shape = r.try_get<PrimitiveShape>(mesh_entity)) r.emplace<PrimitiveShape>(e_new.first, *prim_shape);
    if (r.all_of<SmoothShading>(mesh_entity)) r.emplace<SmoothShading>(e_new.first);
    if (const auto *armature_modifier = r.try_get<ArmatureModifier>(e)) r.emplace<ArmatureModifier>(e_new.second, *armature_modifier);
    if (const auto *bone_attachment = r.try_get<BoneAttachment>(e)) r.emplace<BoneAttachment>(e_new.second, *bone_attachment);
    was_mesh_duplicate = true;
    return e_new.second;
}

entt::entity DuplicateLinkedOne(entt::registry &r, entt::entity e) {
    auto &meshes = r.ctx().get<MeshStore>();
    if (r.all_of<BoneSubPartOf>(e)) return entt::null;
    if (!r.all_of<Instance>(e)) {
        const auto select_behavior = r.all_of<Selected>(e) ? MeshInstanceCreateInfo::SelectBehavior::Additive : MeshInstanceCreateInfo::SelectBehavior::None;

        if (const auto *armature = r.try_get<ArmatureObject>(e)) {
            const auto e_new = r.create();
            r.emplace<Name>(e_new, ::CreateName(r, std::format("{}_copy", GetName(r, e))));
            r.emplace<ObjectKind>(e_new, ObjectType::Armature);
            r.emplace<ArmatureObject>(e_new, armature->Entity);
            const Transform t{r.get<const WorldTransform>(e)};
            r.emplace_or_replace<Transform>(e_new, t);
            r.emplace<WorldTransform>(e_new, t);

            ::ApplySelectBehavior(r, e_new, select_behavior);
            ::CreateBoneInstances(r, meshes, e_new, armature->Entity);
            return e_new;
        }
        return ::AddEmpty(r, meshes, {.Name = std::format("{}_copy", GetName(r, e)), .Transform = Transform{r.get<const WorldTransform>(e)}, .Select = select_behavior});
    }

    const auto mesh_entity = r.get<Instance>(e).Entity;
    const auto e_new = r.create();
    {
        uint instance_count = 0; // Count instances for naming (first duplicated instance is _1, etc.)
        for (const auto [_, instance] : r.view<Instance>().each()) {
            if (instance.Entity == mesh_entity) ++instance_count;
        }
        r.emplace<Name>(e_new, ::CreateName(r, std::format("{}_{}", GetName(r, e), instance_count)));
    }
    r.emplace<Instance>(e_new, mesh_entity);
    r.emplace<ObjectKind>(e_new, ObjectType::Mesh);
    const Transform t_new{r.get<const WorldTransform>(e)};
    r.emplace_or_replace<Transform>(e_new, t_new);
    r.emplace<WorldTransform>(e_new, t_new);
    Show(r, e_new);
    if (const auto *armature_modifier = r.try_get<ArmatureModifier>(e)) r.emplace<ArmatureModifier>(e_new, *armature_modifier);
    if (const auto *bone_attachment = r.try_get<BoneAttachment>(e)) r.emplace<BoneAttachment>(e_new, *bone_attachment);

    r.emplace<Selected>(e_new);

    return e_new;
}
} // namespace

namespace action::object {
void Apply(entt::registry &r, entt::entity viewport, const Action &action) {
    auto &meshes = r.ctx().get<MeshStore>();
    auto begin_translate = [&] { r.emplace_or_replace<StartScreenTransform>(viewport, TransformGizmo::TransformType::Translate); };
    // Hand off Active to the duplicate and drop the source from the selection.
    auto reselect_duplicate = [&](entt::entity src, entt::entity dup) {
        if (r.all_of<Active>(src)) {
            r.remove<Active>(src);
            r.emplace<Active>(dup);
        }
        r.remove<Selected>(src);
    };
    std::visit(
        overloaded{
            [&](Delete) {
                if (!CanDelete(r, viewport)) return;
                for (const auto e : r.view<Selected>(entt::exclude<SubElementOf>) | to<std::vector>()) Destroy(r, viewport, e);
            },
            [&](Duplicate) {
                if (!CanDuplicate(r, viewport)) return;
                const Timer timer{"Duplicate"};
                const auto entities = r.view<Selected>() | to<std::vector>();

                // Pre-reserve arenas to avoid per-CloneMesh buffer growth.
                for (const auto e : entities) {
                    if (!r.all_of<Instance>(e) || r.all_of<BoneSubPartOf>(e)) continue;
                    const auto mesh_entity = r.get<Instance>(e).Entity;
                    if (r.all_of<ObjectExtrasTag>(mesh_entity) || !r.all_of<Mesh>(mesh_entity)) continue;
                    meshes.PlanClone(r.get<const Mesh>(mesh_entity));
                }
                meshes.CommitReserves();

                bool any_mesh_duplicate = false;
                for (const auto e : entities) reselect_duplicate(e, DuplicateOne(r, e, any_mesh_duplicate));
                if (any_mesh_duplicate) r.emplace_or_replace<ProfileNextProcessComponentEvents>(viewport);
                begin_translate();
            },
            [&](DuplicateLinked) {
                if (!CanDuplicateLinked(r, viewport)) return;
                const Timer timer{"DuplicateLinked"};
                for (const auto e : r.view<Selected>() | to<std::vector>()) reselect_duplicate(e, DuplicateLinkedOne(r, e));
                begin_translate();
            },
            [&](ToggleHidden) {
                for (const auto e : r.view<Selected>()) {
                    if (r.all_of<RenderInstance>(e)) Hide(r, e);
                    else Show(r, e);
                }
            },
            [&](const SetSelectedVisible &a) {
                for (const auto e : r.view<const Selected, const Instance>()) {
                    if (r.all_of<SubElementOf>(e)) continue;
                    if (a.Visible) Show(r, e);
                    else Hide(r, e);
                }
            },
            [&](const SetSelectedSmoothShading &a) {
                for (const auto me : ::selection::GetSelectedMeshEntities(r)) {
                    if (r.get<const Mesh>(me).FaceCount() == 0) continue;
                    if (a.Smooth) r.emplace_or_replace<SmoothShading>(me);
                    else r.remove<SmoothShading>(me);
                }
            },
            [&](ParentToActive) {
                const auto active = FindActiveEntity(r);
                if (active == entt::null) return;
                for (const auto e : r.view<Selected>()) {
                    if (e != active) SetParentKeepWorld(r, e, active);
                }
            },
            [&](ClearParent) {
                for (const auto e : r.view<Selected>()) ::ClearParent(r, e);
            },
            [&](const AddEmpty &a) { ::AddEmpty(r, meshes, *a.Info); begin_translate(); },
            [&](const AddArmature &a) {
                const auto &info = *a.Info;
                const auto data_entity = r.create();
                r.emplace<Armature>(data_entity);

                const auto entity = r.create();
                r.emplace<ObjectKind>(entity, ObjectType::Armature);
                r.emplace<ArmatureObject>(entity, data_entity);
                r.emplace<Transform>(entity, info.Transform);
                r.emplace<WorldTransform>(entity, info.Transform);
                r.emplace<Name>(entity, ::CreateName(r, info.Name.empty() ? "Armature" : info.Name));
                ::ApplySelectBehavior(r, entity, info.Select);
                ::CreateBoneInstances(r, meshes, entity, data_entity);
                begin_translate();
            },
            [&](const AddCamera &a) { ::AddCamera(r, meshes, *a.Info, a.Props); begin_translate(); },
            [&](const AddLight &a) { ::AddLight(r, meshes, *a.Info); begin_translate(); },
            [&](const AddMeshPrimitive &a) {
                const auto [mesh_entity, _] = ::AddMesh(r, meshes, meshes.CreateMesh(primitive::CreateMesh(a.Shape), {}, {}), *a.Info);
                r.emplace<PrimitiveShape>(mesh_entity, a.Shape);
                begin_translate();
            },
            [&](const ImportMesh &a) { r.emplace_or_replace<PendingImportMesh>(viewport, a.Path, *a.Info); },
            [&](const ReplaceActive<PrimitiveShape> &a) {
                const auto e = GetActiveMeshEntity(r);
                if (e == entt::null || ::selection::HasScaleLockedInstance(r, e)) return;
                r.emplace_or_replace<PrimitiveShape>(e, a.Value);

                if (auto *mb = r.try_get<MeshBuffers>(e)) ReleaseMeshBuffers(r, *mb);
                r.erase<MeshBuffers>(e);
                r.erase<Mesh>(e);

                auto new_mesh = meshes.CreateMesh(primitive::CreateMesh(a.Value), {}, {});
                r.emplace<MeshBuffers>(e, meshes.GetVerticesRange(new_mesh.GetStoreId()), SlottedRange{}, SlottedRange{}, SlottedRange{});
                r.emplace<Mesh>(e, std::move(new_mesh));
                r.emplace_or_replace<MeshGeometryDirty>(e);
            },
            [&](const SetPbrMeshFeaturesMask &a) {
                const auto e = GetActiveMeshEntity(r);
                if (a.Mask != 0u) r.emplace_or_replace<PbrMeshFeatures>(e, a.Mask);
                else r.remove<PbrMeshFeatures>(e);
            },
            [&]<typename Field>(const Update<Field> &a) { ApplyUpdate(r, viewport, a); },
            [&]<typename T>(const Replace<T> &a) { r.emplace_or_replace<T>(a.Entity, a.Value); },
            [&]<typename T>(const ReplaceActive<T> &a) { r.emplace_or_replace<T>(GetActiveMeshEntity(r), a.Value); },
            [&](const ReplaceActive<PunctualLight> &a) {
                const auto e = FindActiveEntity(r);
                const auto *old = r.try_get<const PunctualLight>(e);
                const auto &n = a.Value;
                if (!old || old->Type != n.Type || old->Range != n.Range || old->OuterConeCos != n.OuterConeCos || old->InnerConeCos != n.InnerConeCos) {
                    r.emplace_or_replace<LightWireframeDirty>(e);
                }
                r.emplace_or_replace<PunctualLight>(e, n);
            },
        },
        action
    );
}
} // namespace action::object
