#include "action/Object.h"
#include "Camera.h"
#include "Profile.h"
#include "Variant.h"
#include "action/Dispatch.h"
#include "action/ScopeResolve.h"
#include "animation/MorphWeights.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "mesh/Primitives.h"
#include "object/ObjectOps.h"
#include "render/GpuBufferOps.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/LightComponents.h"
#include "render/MaterialComponents.h"
#include "render/MeshBuffers.h"
#include "scene/CameraLens.h"
#include "scene/Defaults.h"
#include "scene/SceneGraphOps.h"
#include "scene/WorldTransform.h"
#include "selection/Selection.h"
#include "selection/SelectionGpu.h"
#include "state/Scene.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportEvents.h"

#include <format>
#include <span>

using state::Change;

namespace {
// Rebuild a primitive mesh entity's geometry from its current PrimitiveShape.
void RegeneratePrimitive(state::Scene &r, state::Entity e) {
    const bool was_flat = r.get<const MeshShadingSummary>(e).AllSharp;
    if (auto *mb = r.try_edit<MeshBuffers>(e)) ReleaseMeshBuffers(r, *mb);
    // Erasing MeshHandle fires on_destroy, releasing the old store entry.
    r.remove<MeshBuffers, MeshHandle>(e);
    const auto created = CreateMesh(r, {.Data = primitive::CreateMesh(r.get<const PrimitiveShape>(e)), .FlatShaded = was_flat});
    r.emplace<MeshHandle>(e, MeshHandle{created.StoreId});
    r.emplace_or_replace<MeshGeometryDirty>(e);
}

// Create an armature object over `data_entity`, creating fresh armature data when null.
state::Entity CreateArmatureObject(state::Scene &r, MeshStore &meshes, state::Entity data_entity, std::string_view name, const Transform &transform, MeshInstanceCreateInfo::SelectBehavior select) {
    if (data_entity == state::Null) {
        data_entity = r.create();
        r.emplace<Armature>(data_entity);
    }
    const auto entity = r.create();
    r.emplace<ObjectKind>(entity, ObjectType::Armature);
    r.emplace<ArmatureObject>(entity, data_entity);
    r.emplace<Transform>(entity, transform);
    EmplaceUniqueName(r, entity, name.empty() ? "Armature" : name);
    ::ApplySelectBehavior(r, entity, select);
    ::CreateBoneInstances(r, meshes, entity, data_entity);
    return entity;
}

state::Entity DuplicateOne(state::Scene &r, state::Entity e) {
    auto &meshes = r.ctx().get<MeshStore>();
    const ObjectCreateInfo create_info{
        .Name = std::format("{}_copy", GetName(r, e)),
        // Duplicate is created at root, so its local must match source's world.
        .Transform = Transform{r.get<const WorldTransform>(e)},
        .Select = r.all_of<Selected>(e) ? MeshInstanceCreateInfo::SelectBehavior::Additive : MeshInstanceCreateInfo::SelectBehavior::None,
    };

    if (!r.all_of<Instance>(e)) {
        if (const auto object_type = r.all_of<ObjectKind>(e) ? r.get<const ObjectKind>(e).Value : ObjectType::Empty; object_type == ObjectType::Armature) {
            const auto copy_entity = CreateArmatureObject(r, meshes, state::Null, create_info.Name, create_info.Transform, create_info.Select);
            if (const auto *src_armature = r.try_get<ArmatureObject>(e)) {
                r.edit<Armature>(r.get<const ArmatureObject>(copy_entity).Entity) = r.get<const Armature>(src_armature->Entity);
            }
            return copy_entity;
        }
        return ::AddEmpty(r, meshes, create_info);
    }

    // Bone sub-entities (head/tail joints, bone instances) are not independently duplicable.
    if (r.all_of<BoneSubPartOf>(e)) return state::Null;

    // Object extras (Camera, Empty, Light) have Instance but create their own wireframe mesh.
    if (r.all_of<ObjectExtrasTag>(r.get<Instance>(e).Entity)) {
        if (const auto lens = LensOf(r, e)) return ::AddCamera(r, meshes, create_info, *lens);
        if (const auto *light = r.try_get<const PunctualLight>(e)) return ::AddLight(r, meshes, create_info, *light);
        return ::AddEmpty(r, meshes, create_info);
    }

    const auto mesh_entity = r.get<Instance>(e).Entity;
    const auto e_new = ::AddMesh(
        r, meshes.CloneMesh(GetMesh(r, mesh_entity)),
        MeshInstanceCreateInfo{.Name = create_info.Name, .Transform = create_info.Transform, .Select = create_info.Select, .Visible = r.all_of<RenderInstance>(e)}
    );
    if (auto *prim_shape = r.try_get<PrimitiveShape>(mesh_entity)) r.emplace<PrimitiveShape>(e_new.first, *prim_shape);
    if (const auto *armature_modifier = r.try_get<ArmatureModifier>(e)) r.emplace<ArmatureModifier>(e_new.second, *armature_modifier);
    if (const auto *bone_attachment = r.try_get<BoneAttachment>(e)) r.emplace<BoneAttachment>(e_new.second, *bone_attachment);
    if (const auto *weights = r.try_get<const MorphWeightRange>(e)) r.emplace<MorphWeightRange>(e_new.second, r.ctx().get<GpuBuffers>().MorphWeightBuffer.Clone(weights->Weights));
    return e_new.second;
}

state::Entity DuplicateLinkedOne(state::Scene &r, state::Entity e) {
    auto &meshes = r.ctx().get<MeshStore>();
    if (r.all_of<BoneSubPartOf>(e)) return state::Null;
    if (!r.all_of<Instance>(e)) {
        const auto select_behavior = r.all_of<Selected>(e) ? MeshInstanceCreateInfo::SelectBehavior::Additive : MeshInstanceCreateInfo::SelectBehavior::None;

        if (const auto *armature = r.try_get<ArmatureObject>(e)) {
            return CreateArmatureObject(r, meshes, armature->Entity, std::format("{}_copy", GetName(r, e)), Transform{r.get<const WorldTransform>(e)}, select_behavior);
        }
        return ::AddEmpty(r, meshes, {.Name = std::format("{}_copy", GetName(r, e)), .Transform = Transform{r.get<const WorldTransform>(e)}, .Select = select_behavior});
    }

    const auto mesh_entity = r.get<Instance>(e).Entity;
    const auto e_new = r.create();
    {
        uint32_t instance_count{0}; // Count instances for naming (first duplicated instance is _1, etc.)
        for (const auto [_, instance] : r.view<Instance>().each()) {
            if (instance.Entity == mesh_entity) ++instance_count;
        }
        EmplaceUniqueName(r, e_new, std::format("{}_{}", GetName(r, e), instance_count));
    }
    r.emplace<Instance>(e_new, mesh_entity);
    r.emplace<ObjectKind>(e_new, ObjectType::Mesh);
    const Transform t_new{r.get<const WorldTransform>(e)};
    r.emplace_or_replace<Transform>(e_new, t_new);
    Show(r, e_new);
    if (const auto *armature_modifier = r.try_get<ArmatureModifier>(e)) r.emplace<ArmatureModifier>(e_new, *armature_modifier);
    if (const auto *bone_attachment = r.try_get<BoneAttachment>(e)) r.emplace<BoneAttachment>(e_new, *bone_attachment);
    if (const auto *weights = r.try_get<const MorphWeightRange>(e)) r.emplace<MorphWeightRange>(e_new, r.ctx().get<GpuBuffers>().MorphWeightBuffer.Clone(weights->Weights));

    r.emplace<Selected>(e_new);

    return e_new;
}
} // namespace

namespace action {
bool UpdateTraits<PrimitiveShape>::Has(const state::Scene &r, state::Entity e) { return r.all_of<PrimitiveShape>(e); }
state::Entity UpdateTraits<PrimitiveShape>::Active(const state::Scene &r) {
    const auto e = GetActiveMeshEntity(r);
    return e != state::Null && Has(r, e) ? e : state::Null;
}
// Only meshes of the active primitive's kind share its fields.
void UpdateTraits<PrimitiveShape>::ForEachSelected(state::Scene &r, const std::function<void(state::Entity)> &fn) {
    const auto active = Active(r);
    if (active == state::Null) return;
    const auto kind = r.get<const PrimitiveShape>(active).index();
    for (const auto e : ::selection::GetSelectedMeshEntities(r))
        if (Has(r, e) && r.get<const PrimitiveShape>(e).index() == kind) fn(e);
}
void UpdateTraits<PrimitiveShape>::Read(const state::Scene &r, state::Entity e, uint16_t offset, void *dst, size_t size) {
    std::visit([&](const auto &alt) { std::memcpy(dst, reinterpret_cast<const std::byte *>(&alt) + offset, size); }, r.get<const PrimitiveShape>(e));
}
void UpdateTraits<PrimitiveShape>::Write(state::Scene &r, state::Entity e, uint16_t offset, const void *src, size_t size) {
    if (::selection::HasScaleLockedInstance(r, e)) return;
    r.patch<PrimitiveShape>(e, [&](PrimitiveShape &s) { std::visit([&](auto &alt) { std::memcpy(reinterpret_cast<std::byte *>(&alt) + offset, src, size); }, s); });
    RegeneratePrimitive(r, e);
}
} // namespace action

namespace action::object {
void Apply(state::Scene &r, state::Entity viewport, const Action &action) {
    auto &meshes = r.ctx().get<MeshStore>();
    auto begin_translate = [&] { r.emplace_or_replace<StartScreenTransform>(viewport, TransformGizmo::TransformType::Translate); };
    const auto duplicate = [&](bool linked, const PendingTransform *placement = nullptr) {
        if (!(linked ? CanDuplicateLinked(r, viewport) : CanDuplicate(r, viewport))) return;
        const profile::CpuScope scope{linked ? "DuplicateLinked" : "Duplicate"};
        const auto entities = SortedEntities(r.view<Selected>());
        if (!linked) {
            // Pre-reserve arenas to avoid per-CloneMesh buffer growth.
            for (const auto e : entities) {
                if (r.all_of<Instance>(e) && !r.all_of<BoneSubPartOf>(e)) {
                    const auto mesh_entity = r.get<Instance>(e).Entity;
                    if (!r.all_of<ObjectExtrasTag>(mesh_entity) && HasMesh(r, mesh_entity)) meshes.PlanClone(GetMesh(r, mesh_entity));
                }
            }
            meshes.CommitReserves();
        }
        for (const auto src : entities) {
            const auto dup = linked ? DuplicateLinkedOne(r, src) : DuplicateOne(r, src);
            // Copies are rooted in world space, so placement needs no parent conversion.
            if (placement) r.patch<Transform>(dup, [&](auto &t) { t = placement->ApplyTo(t, r.all_of<ScaleLocked>(dup)); });
            if (r.all_of<Active>(src)) {
                r.remove<Active>(src);
                r.emplace<Active>(dup);
            }
            r.remove<Selected>(src);
        }
        if (placement) r.remove<StartScreenTransform>(viewport);
        else begin_translate();
    };
    // Mesh-data components live on the object's mesh entity.
    auto for_each_mesh_target = [&](Scope scope, auto &&fn) {
        ForEachScopeTarget(
            scope, state::Null, state::Null,
            [&] { return GetActiveMeshEntity(r); },
            [&](auto &&f) { for (const auto e : ::selection::GetSelectedMeshEntities(r)) f(e); },
            fn
        );
    };
    const auto edit_selection_meshes = [&](Element element) {
        std::vector<state::Entity> result;
        if (r.get<const EditMode>(viewport).Value != element) return result;
        for (const auto me : r.view<const MeshElementSelection>()) {
            if (!HasMesh(r, me)) continue;
            const auto mesh = GetMesh(r, me);
            const auto &summary = meshes.GetSelectionSummary(mesh.GetStoreId());
            if (summary.Mode == element && summary.SelectedCount > 0) result.push_back(me);
        }
        return result;
    };
    const auto selected_meshes = [&] {
        const auto selected = ::selection::GetSelectedMeshEntities(r);
        return std::vector<state::Entity>{selected.begin(), selected.end()};
    };
    std::visit(
        overloaded{
            [&](Delete) {
                if (!CanDelete(r, viewport)) return;
                for (const auto e : SortedEntities(r.view<Selected>(state::Exclude<SubElementOf>))) Destroy(r, viewport, e);
            },
            [&](Duplicate) { duplicate(false); },
            [&](DuplicateLinked) { duplicate(true); },
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
                const auto targets = selected_meshes();
                ApplyEditSharpness(
                    r, viewport, targets,
                    a.Smooth ? EditSharpnessOperation::SmoothAll : EditSharpnessOperation::SetAllFaces,
                    !a.Smooth
                );
            },
            [&](const ShadeSelectedSmoothByAngle &a) {
                const auto targets = selected_meshes();
                ApplyEditSharpness(r, viewport, targets, EditSharpnessOperation::SmoothByAngle, false, a.Angle);
            },
            [&](const SetSelectedSharp &a) {
                const auto targets = edit_selection_meshes(a.Element);
                const auto operation = a.Element == Element::Face ? EditSharpnessOperation::SetSelectedFaces :
                    a.Element == Element::Edge                    ? EditSharpnessOperation::SetSelectedEdges :
                                                                    EditSharpnessOperation::SetVertexEdges;
                ApplyEditSharpness(r, viewport, targets, operation, a.Sharp);
            },
            [&](ParentToActive) {
                const auto active = FindActiveEntity(r);
                if (active == state::Null) return;
                for (const auto e : r.view<Selected>()) {
                    if (e != active) SetParentKeepWorld(r, e, active);
                }
            },
            [&](ClearParent) {
                for (const auto e : r.view<Selected>()) ::ClearParent(r, e);
            },
            [&](const AddEmpty &a) { ::AddEmpty(r, meshes, *a.Info); begin_translate(); },
            [&](const AddArmature &a) {
                CreateArmatureObject(r, meshes, state::Null, a.Info->Name, a.Info->Transform, a.Info->Select);
                begin_translate();
            },
            [&](const AddCamera &a) { ::AddCamera(r, meshes, *a.Info, a.Props); begin_translate(); },
            [&](const AddLight &a) { ::AddLight(r, meshes, *a.Info); begin_translate(); },
            [&](const AddMeshPrimitive &a) {
                const auto created = CreateMesh(r, {.Data = primitive::CreateMesh(a.Shape), .FlatShaded = true});
                const auto [mesh_entity, _] = ::AddMesh(r, created.StoreId, *a.Info);
                r.emplace<PrimitiveShape>(mesh_entity, a.Shape);
                begin_translate();
            },
            [&](const ImportMesh &a) { RequestImportMesh(r, viewport, a.Path, *a.Info); },
            [&](const SetPbrMeshFeaturesMask &a) {
                for_each_mesh_target(a.Scope, [&](state::Entity e) {
                    if (a.Mask != 0u) r.emplace_or_replace<PbrMeshFeatures>(e, a.Mask);
                    else r.remove<PbrMeshFeatures>(e);
                });
            },
            [&]<typename T>(const UpdateMaterial<T> &a) {
                auto &materials = r.ctx().get<GpuBuffers>().Materials;
                if (a.Index >= materials.Count<PBRMaterial>()) return;
                materials.Update(std::as_bytes(std::span{&a.Value, 1}), uint64_t(a.Index) * sizeof(PBRMaterial) + a.Offset);
                reactive(r, Change::Materials).emplace(viewport);
            },
            [&](const SetMaterialSlotSelection &a) {
                for_each_mesh_target(a.Scope, [&](state::Entity e) { r.emplace_or_replace<MeshMaterialSlotSelection>(e, a.PrimitiveIndex); });
            },
            [&](const SetMaterialAssignment &a) {
                for_each_mesh_target(a.Scope, [&](state::Entity e) { r.emplace_or_replace<MeshMaterialAssignment>(e, a.PrimitiveIndex, a.MaterialIndex); });
            },
            [&](const SetProjection &a) {
                ForEachScopeTarget(
                    a.Scope, state::Null, state::Null,
                    [&] { const auto e = FindActiveEntity(r); return HasLens(r, e) ? e : state::Null; },
                    [&](auto &&fn) {
                        for (const auto e : r.view<Selected>())
                            if (HasLens(r, e)) fn(e);
                    },
                    [&](state::Entity e) {
                        const float distance = std::max(numeric::Length(r.get<const WorldTransform>(e).P), 1.f);
                        if (const auto *perspective = r.try_get<const Perspective>(e); perspective && a.Orthographic) SetLens(r, e, OrthographicFromPerspective(*perspective, distance));
                        else if (const auto *orthographic = r.try_get<const Orthographic>(e); orthographic && !a.Orthographic) SetLens(r, e, PerspectiveFromOrthographic(*orthographic, distance));
                    }
                );
            },
            [&](const SetLightType &a) {
                ForEachComponentTarget<PunctualLight>(r, a.Scope, state::Null, state::Null, [&](auto e) {
                    r.patch<PunctualLight>(e, [&](auto &light) {
                        auto next = Defaults::MakePunctualLight(a.Type);
                        next.Color = light.Color;
                        next.Intensity = light.Intensity;
                        light = next;
                    });
                });
            },
            [&](const SetSpotCone &a) {
                ForEachComponentTarget<PunctualLight>(r, a.Scope, state::Null, state::Null, [&](auto e) {
                    r.patch<PunctualLight>(e, [&](auto &light) {
                        light.OuterConeAngle = a.OuterAngle;
                        light.InnerConeAngle = a.OuterAngle * (1.f - a.Blend);
                    });
                });
            },
        },
        action
    );
}
} // namespace action::object
