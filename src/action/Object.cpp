#include "action/Object.h"
#include "Profile.h"
#include "action/Dispatch.h"
#include "action/ScopeResolve.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "mesh/MeshBatch.h"
#include "mesh/MeshComponents.h"
#include "mesh/Primitives.h"
#include "object/ObjectOps.h"
#include "project/Registry.h"
#include "render/GpuBufferOps.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/LightComponents.h"
#include "render/MeshBuffers.h"
#include "scene/Defaults.h"
#include "scene/SceneGraphOps.h"
#include "scene/WorldTransform.h"
#include "selection/Selection.h"
#include "selection/SelectionQueries.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportEvents.h"

#include <format>

namespace {
// Read/write a field at `offset` within a PrimitiveShape's current alternative.
void ReadPrimitiveField(const entt::registry &r, entt::entity e, uint16_t offset, void *dst, uint16_t size) {
    std::visit([&](const auto &alt) { std::memcpy(dst, reinterpret_cast<const std::byte *>(&alt) + offset, size); }, r.get<const PrimitiveShape>(e));
}
void PatchPrimitiveField(entt::registry &r, entt::entity e, uint16_t offset, const void *src, uint16_t size) {
    project::Patch<PrimitiveShape>(r, e, [&](PrimitiveShape &s) { std::visit([&](auto &alt) { std::memcpy(reinterpret_cast<std::byte *>(&alt) + offset, src, size); }, s); });
}
inline const action::detail::ComponentPatcher PrimitiveFieldPatcher{&PatchPrimitiveField, &ReadPrimitiveField, &action::detail::HasComponent<PrimitiveShape>, "PrimitiveShape"};

// Create an armature object over `data_entity`, creating fresh armature data when null.
entt::entity CreateArmatureObject(entt::registry &r, MeshStore &meshes, entt::entity data_entity, std::string_view name, const Transform &transform, MeshInstanceCreateInfo::SelectBehavior select) {
    if (data_entity == entt::null) {
        data_entity = project::Create(r);
        project::Emplace<Armature>(r, data_entity);
    }
    const auto entity = project::Create(r);
    project::Emplace<ObjectKind>(r, entity, ObjectType::Armature);
    project::Emplace<ArmatureObject>(r, entity, data_entity);
    project::Emplace<Transform>(r, entity, transform);
    EmplaceUniqueName(r, entity, name.empty() ? "Armature" : name);
    ::ApplySelectBehavior(r, entity, select);
    ::CreateBoneInstances(r, meshes, entity, data_entity);
    return entity;
}

entt::entity DuplicateOne(entt::registry &r, entt::entity e) {
    auto &meshes = r.ctx().get<MeshStore>();
    const ObjectCreateInfo create_info{
        .Name = std::format("{}_copy", GetName(r, e)),
        // Duplicate is created at root, so its local must match source's world.
        .Transform = Transform{r.get<const WorldTransform>(e)},
        .Select = r.all_of<Selected>(e) ? MeshInstanceCreateInfo::SelectBehavior::Additive : MeshInstanceCreateInfo::SelectBehavior::None,
    };

    if (!r.all_of<Instance>(e)) {
        if (const auto object_type = r.all_of<ObjectKind>(e) ? r.get<const ObjectKind>(e).Value : ObjectType::Empty; object_type == ObjectType::Armature) {
            const auto copy_entity = CreateArmatureObject(r, meshes, entt::null, create_info.Name, create_info.Transform, create_info.Select);
            if (const auto *src_armature = r.try_get<ArmatureObject>(e)) {
                r.get<Armature>(r.get<const ArmatureObject>(copy_entity).Entity) = r.get<const Armature>(src_armature->Entity);
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
        r, meshes.CloneMesh(GetMesh(r, mesh_entity)).StoreId,
        MeshInstanceCreateInfo{.Name = create_info.Name, .Transform = create_info.Transform, .Select = create_info.Select, .Visible = r.all_of<RenderInstance>(e)}
    );
    if (auto *prim_shape = r.try_get<PrimitiveShape>(mesh_entity)) project::Emplace<PrimitiveShape>(r, e_new.first, *prim_shape);
    if (const auto *armature_modifier = r.try_get<ArmatureModifier>(e)) project::Emplace<ArmatureModifier>(r, e_new.second, *armature_modifier);
    if (const auto *bone_attachment = r.try_get<BoneAttachment>(e)) project::Emplace<BoneAttachment>(r, e_new.second, *bone_attachment);
    return e_new.second;
}

entt::entity DuplicateLinkedOne(entt::registry &r, entt::entity e) {
    auto &meshes = r.ctx().get<MeshStore>();
    if (r.all_of<BoneSubPartOf>(e)) return entt::null;
    if (!r.all_of<Instance>(e)) {
        const auto select_behavior = r.all_of<Selected>(e) ? MeshInstanceCreateInfo::SelectBehavior::Additive : MeshInstanceCreateInfo::SelectBehavior::None;

        if (const auto *armature = r.try_get<ArmatureObject>(e)) {
            return CreateArmatureObject(r, meshes, armature->Entity, std::format("{}_copy", GetName(r, e)), Transform{r.get<const WorldTransform>(e)}, select_behavior);
        }
        return ::AddEmpty(r, meshes, {.Name = std::format("{}_copy", GetName(r, e)), .Transform = Transform{r.get<const WorldTransform>(e)}, .Select = select_behavior});
    }

    const auto mesh_entity = r.get<Instance>(e).Entity;
    const auto e_new = project::Create(r);
    {
        uint32_t instance_count{0}; // Count instances for naming (first duplicated instance is _1, etc.)
        for (const auto [_, instance] : r.view<Instance>().each()) {
            if (instance.Entity == mesh_entity) ++instance_count;
        }
        EmplaceUniqueName(r, e_new, std::format("{}_{}", GetName(r, e), instance_count));
    }
    project::Emplace<Instance>(r, e_new, mesh_entity);
    project::Emplace<ObjectKind>(r, e_new, ObjectType::Mesh);
    const Transform t_new{r.get<const WorldTransform>(e)};
    project::EmplaceOrReplace<Transform>(r, e_new, t_new);
    Show(r, e_new);
    if (const auto *armature_modifier = r.try_get<ArmatureModifier>(e)) project::Emplace<ArmatureModifier>(r, e_new, *armature_modifier);
    if (const auto *bone_attachment = r.try_get<BoneAttachment>(e)) project::Emplace<BoneAttachment>(r, e_new, *bone_attachment);

    project::Emplace<Selected>(r, e_new);

    return e_new;
}
} // namespace

namespace action::object {
void Apply(entt::registry &r, entt::entity viewport, const Action &action) {
    auto &meshes = r.ctx().get<MeshStore>();
    auto begin_translate = [&] { project::EmplaceOrReplace<StartScreenTransform>(r, viewport, TransformGizmo::TransformType::Translate); };
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
            if (placement) project::Patch<Transform>(r, dup, [&](auto &t) { t = placement->ApplyTo(t, r.all_of<ScaleLocked>(dup)); });
            if (r.all_of<Active>(src)) {
                project::Remove<Active>(r, src);
                project::Emplace<Active>(r, dup);
            }
            project::Remove<Selected>(r, src);
        }
        if (placement) project::Remove<StartScreenTransform>(r, viewport);
        else begin_translate();
    };
    // Rebuild a primitive mesh entity's geometry from its current PrimitiveShape.
    auto regen_primitive = [&](entt::entity e) {
        const bool was_flat = r.get<const MeshShadingSummary>(e).AllSharp;
        if (auto *mb = r.try_get<MeshBuffers>(e)) ReleaseMeshBuffers(r, *mb);
        // Erasing MeshHandle fires on_destroy, releasing the old store entry.
        project::Erase<MeshBuffers, MeshHandle>(r, e);
        const auto created = CreateMesh(r, {.Data = primitive::CreateMesh(r.get<const PrimitiveShape>(e)), .FlatShaded = was_flat});
        project::Emplace<MeshHandle>(r, e, MeshHandle{created.StoreId});
        project::EmplaceOrReplace<MeshGeometryDirty>(r, e);
    };
    auto for_each_mesh_target = [&](Scope scope, entt::entity entity, auto &&fn) {
        switch (scope) {
            case Scope::Entity: fn(entity); break;
            case Scope::Active:
                if (const auto e = GetActiveMeshEntity(r); e != entt::null) fn(e);
                break;
            case Scope::Selected:
            case Scope::SelectedDelta:
                for (const auto e : ::selection::GetSelectedMeshEntities(r)) fn(e);
                break;
        }
    };
    const auto edit_selection_meshes = [&](Element element) {
        std::vector<entt::entity> result;
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
        return std::vector<entt::entity>{selected.begin(), selected.end()};
    };
    std::visit(
        overloaded{
            [&](Delete) {
                if (!CanDelete(r, viewport)) return;
                for (const auto e : SortedEntities(r.view<Selected>(entt::exclude<SubElementOf>))) Destroy(r, viewport, e);
            },
            [&](Duplicate) { duplicate(false); },
            [&](DuplicateLinked) { duplicate(true); },
            [&](const DuplicateToPosition &a) { duplicate(a.Linked, a.Placement.get()); },
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
            [&](const SetSelectedFacesSmooth &a) {
                const auto targets = edit_selection_meshes(Element::Face);
                ApplyEditSharpness(r, viewport, targets, EditSharpnessOperation::SetSelectedFaces, !a.Smooth);
            },
            [&](const SetSelectedEdgesSharp &a) {
                const auto targets = edit_selection_meshes(Element::Edge);
                ApplyEditSharpness(r, viewport, targets, EditSharpnessOperation::SetSelectedEdges, a.Sharp);
            },
            [&](const SetSelectedVertexEdgesSharp &a) {
                const auto targets = edit_selection_meshes(Element::Vertex);
                ApplyEditSharpness(r, viewport, targets, EditSharpnessOperation::SetVertexEdges, a.Sharp);
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
                CreateArmatureObject(r, meshes, entt::null, a.Info->Name, a.Info->Transform, a.Info->Select);
                begin_translate();
            },
            [&](const AddCamera &a) { ::AddCamera(r, meshes, *a.Info, a.Props); begin_translate(); },
            [&](const AddLight &a) { ::AddLight(r, meshes, *a.Info); begin_translate(); },
            [&](const AddMeshPrimitive &a) {
                const auto created = CreateMesh(r, {.Data = primitive::CreateMesh(a.Shape), .FlatShaded = true});
                const auto [mesh_entity, _] = ::AddMesh(r, created.StoreId, *a.Info);
                project::Emplace<PrimitiveShape>(r, mesh_entity, a.Shape);
                begin_translate();
            },
            [&](const ImportMesh &a) { RequestImportMesh(r, viewport, a.Path, *a.Info); },
            [&]<typename Field>(const UpdatePrimitiveField<Field> &a) {
                static const auto comp = entt::type_hash<PrimitiveShape>::value();
                const auto active = GetActiveMeshEntity(r);
                if (active == entt::null || !r.all_of<PrimitiveShape>(active)) return;
                const auto active_index = r.get<const PrimitiveShape>(active).index();
                auto clamp_field = [&](Field v) {
                    if constexpr (std::integral<Field>) return std::clamp(v, a.Min, a.Max);
                    else return numeric::Clamp(v, a.Min, a.Max);
                };
                auto write = [&](entt::entity e, Field value) {
                    if (::selection::HasScaleLockedInstance(r, e)) return;
                    value = clamp_field(value);
                    PrimitiveFieldPatcher.Patch(r, e, a.Offset, &value, sizeof(Field));
                    regen_primitive(e);
                };
                if (a.Scope == Scope::SelectedDelta) {
                    // Add the active's delta to each member's own start, keeping their relative values.
                    const auto active_start = FieldGestureStart<Field>(r, active, PrimitiveFieldPatcher, comp, a.Offset);
                    for (const auto e : ::selection::GetSelectedMeshEntities(r)) {
                        if (!r.all_of<PrimitiveShape>(e) || r.get<const PrimitiveShape>(e).index() != active_index) continue;
                        const auto e_start = FieldGestureStart<Field>(r, e, PrimitiveFieldPatcher, comp, a.Offset);
                        if constexpr (std::integral<Field>) {
                            // Accumulate in a wider signed type so an unsigned field can't wrap on a downward delta.
                            write(e, Field(std::clamp<int64_t>(int64_t(e_start) + int64_t(a.Value) - int64_t(active_start), int64_t(a.Min), int64_t(a.Max))));
                        } else {
                            write(e, e_start + (a.Value - active_start));
                        }
                    }
                } else if (a.Scope == Scope::Selected) {
                    for (const auto e : ::selection::GetSelectedMeshEntities(r)) {
                        if (r.all_of<PrimitiveShape>(e) && r.get<const PrimitiveShape>(e).index() == active_index) write(e, a.Value);
                    }
                } else {
                    write(active, a.Value);
                }
            },
            [&](const SetPbrMeshFeaturesMask &a) {
                for_each_mesh_target(a.Scope, entt::null, [&](entt::entity e) {
                    if (a.Mask != 0u) project::EmplaceOrReplace<PbrMeshFeatures>(r, e, a.Mask);
                    else project::Remove<PbrMeshFeatures>(r, e);
                });
            },
            [&](const UpdateMaterial &a) {
                if (a.Features) Apply(r, viewport, SetPbrMeshFeaturesMask{*a.Features, a.Scope});
                r.ctx().get<GpuBuffers>().Materials.Set(a.Index, *a.Value);
                project::EmplaceOrReplace<MaterialDirty>(r, viewport, a.Index);
            },
            [&]<typename Field>(const Update<Field> &a) { ApplyUpdate(r, viewport, a); },
            // Mesh-data components (material assignment / slot selection) live on the object's mesh entity.
            [&]<typename T>(const Replace<T> &a) { for_each_mesh_target(a.Scope, a.Entity, [&](entt::entity e) { project::EmplaceOrReplace<T>(r, e, a.Value); }); },
            [&](const SetLightType &a) {
                ForEachReplaceTarget<PunctualLight>(r, a.Scope, entt::null, [&](auto e) {
                    project::Patch<PunctualLight>(r, e, [&](auto &light) {
                        auto next = Defaults::MakePunctualLight(a.Type);
                        next.Color = light.Color;
                        next.Intensity = light.Intensity;
                        light = next;
                    });
                });
            },
            [&](const SetSpotCone &a) {
                ForEachReplaceTarget<PunctualLight>(r, a.Scope, entt::null, [&](auto e) {
                    project::Patch<PunctualLight>(r, e, [&](auto &light) {
                        light.OuterConeCos = std::cos(a.OuterAngle);
                        light.InnerConeCos = std::cos(a.OuterAngle * (1.f - a.Blend));
                    });
                });
            },
        },
        action
    );
}
} // namespace action::object
