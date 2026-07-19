#include "action/Object.h"
#include "action/Dispatch.h"
#include "action/ScopeResolve.h"
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
#include "render/Profile.h"
#include "scene/SceneGraphOps.h"
#include "scene/WorldTransform.h"
#include "selection/Selection.h"
#include "selection/SelectionBitset.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"

using std::ranges::to;

namespace {
// Read/write a field at `offset` within a PrimitiveShape's current alternative.
void ReadPrimitiveField(const entt::registry &r, entt::entity e, uint16_t offset, void *dst, uint16_t size) {
    std::visit([&](const auto &alt) { std::memcpy(dst, reinterpret_cast<const std::byte *>(&alt) + offset, size); }, r.get<const PrimitiveShape>(e));
}
void PatchPrimitiveField(entt::registry &r, entt::entity e, uint16_t offset, const void *src, uint16_t size) {
    r.patch<PrimitiveShape>(e, [&](PrimitiveShape &s) { std::visit([&](auto &alt) { std::memcpy(reinterpret_cast<std::byte *>(&alt) + offset, src, size); }, s); });
}
inline const action::detail::ComponentPatcher PrimitiveFieldPatcher{&PatchPrimitiveField, &ReadPrimitiveField, &action::detail::HasComponent<PrimitiveShape>, "PrimitiveShape"};

// Create an armature object over `data_entity`, creating fresh armature data when null.
entt::entity CreateArmatureObject(entt::registry &r, MeshStore &meshes, entt::entity data_entity, std::string_view name, const Transform &transform, MeshInstanceCreateInfo::SelectBehavior select) {
    if (data_entity == entt::null) {
        data_entity = r.create();
        r.emplace<Armature>(data_entity);
    }
    const auto entity = r.create();
    r.emplace<ObjectKind>(entity, ObjectType::Armature);
    r.emplace<ArmatureObject>(entity, data_entity);
    r.emplace<Transform>(entity, transform);
    r.emplace<Name>(entity, ::CreateName(r, name.empty() ? "Armature" : name));
    ::ApplySelectBehavior(r, entity, select);
    ::CreateBoneInstances(r, meshes, entity, data_entity);
    return entity;
}

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
        r, meshes,
        meshes.CloneMesh(GetMesh(r, mesh_entity)),
        MeshInstanceCreateInfo{.Name = create_info.Name, .Transform = create_info.Transform, .Select = create_info.Select, .Visible = r.all_of<RenderInstance>(e)}
    );
    if (auto *prim_shape = r.try_get<PrimitiveShape>(mesh_entity)) r.emplace<PrimitiveShape>(e_new.first, *prim_shape);
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
        r.emplace<Name>(e_new, ::CreateName(r, std::format("{}_{}", GetName(r, e), instance_count)));
    }
    r.emplace<Instance>(e_new, mesh_entity);
    r.emplace<ObjectKind>(e_new, ObjectType::Mesh);
    const Transform t_new{r.get<const WorldTransform>(e)};
    r.emplace_or_replace<Transform>(e_new, t_new);
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
    // Rebuild a primitive mesh entity's geometry from its current PrimitiveShape.
    auto regen_primitive = [&](entt::entity e) {
        const bool was_flat = meshes.GetFaceSharpnessSummary(r.get<const MeshHandle>(e).StoreId).All;
        if (auto *mb = r.try_get<MeshBuffers>(e)) ReleaseMeshBuffers(r, *mb);
        // Erasing MeshHandle fires on_destroy, releasing the old store entry.
        r.erase<MeshBuffers, MeshHandle, MeshConnectivity>(e);
        auto new_mesh = meshes.CreateMesh(primitive::CreateMesh(r.get<const PrimitiveShape>(e)), {}, {}, was_flat);
        r.emplace<MeshConnectivity>(e, std::move(new_mesh.Connectivity));
        r.emplace<MeshHandle>(e, MeshHandle{new_mesh.StoreId});
        r.emplace_or_replace<MeshGeometryDirty>(e);
    };
    // `fn` for each mesh entity a scope targets (the carried entity, the active mesh, or each selected mesh).
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
    // `fn(mesh_entity, mesh, bits, range)` for each edit-mode mesh with a non-empty element selection, when `element` is the current edit mode.
    auto for_each_edit_selection = [&](Element element, auto &&fn) {
        if (r.get<const EditMode>(viewport).Value != element) return;
        const auto *bits_ref = r.ctx().find<const SelectionBitsetRef>();
        if (!bits_ref) return;
        for (const auto [me, br] : r.view<const MeshSelectionBitsetRange>().each()) {
            if (!HasMesh(r, me)) continue;
            if (::selection::CountSelected(bits_ref->Value.data(), br.Offset, br.Count) == 0) continue;
            fn(me, GetMesh(r, me), bits_ref->Value.data(), br);
        }
    };
    std::visit(
        overloaded{
            [&](Delete) {
                if (!CanDelete(r, viewport)) return;
                for (const auto e : r.view<Selected>(entt::exclude<SubElementOf>) | to<std::vector>()) Destroy(r, viewport, e);
            },
            [&](Duplicate) {
                if (!CanDuplicate(r, viewport)) return;
                const profile::CpuScope scope{"Duplicate"};
                const auto entities = r.view<Selected>() | to<std::vector>();

                // Pre-reserve arenas to avoid per-CloneMesh buffer growth.
                for (const auto e : entities) {
                    if (r.all_of<Instance>(e) && !r.all_of<BoneSubPartOf>(e)) {
                        const auto mesh_entity = r.get<Instance>(e).Entity;
                        if (!r.all_of<ObjectExtrasTag>(mesh_entity) && HasMesh(r, mesh_entity)) meshes.PlanClone(GetMesh(r, mesh_entity));
                    }
                }
                meshes.CommitReserves();

                bool any_mesh_duplicate = false;
                for (const auto e : entities) reselect_duplicate(e, DuplicateOne(r, e, any_mesh_duplicate));
                begin_translate();
            },
            [&](DuplicateLinked) {
                if (!CanDuplicateLinked(r, viewport)) return;
                const profile::CpuScope scope{"DuplicateLinked"};
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
                    const auto mesh = GetMesh(r, me);
                    if (mesh.FaceCount() == 0) continue;
                    std::ranges::fill(meshes.GetFaceSharpness(mesh.GetStoreId()), uint8_t(a.Smooth ? 0 : 1));
                    // Shade Smooth clears edge sharpness.
                    if (a.Smooth) std::ranges::fill(meshes.GetEdgeSharpness(mesh.GetStoreId()), uint8_t{0});
                    r.emplace_or_replace<MeshShadingDirty>(me);
                }
            },
            [&](const ShadeSelectedSmoothByAngle &a) {
                for (const auto me : ::selection::GetSelectedMeshEntities(r)) {
                    const auto mesh = GetMesh(r, me);
                    if (mesh.FaceCount() == 0) continue;
                    std::ranges::fill(meshes.GetFaceSharpness(mesh.GetStoreId()), uint8_t{0});
                    meshes.SetEdgeSharpnessByAngle(mesh, a.Angle);
                    r.emplace_or_replace<MeshShadingDirty>(me);
                }
            },
            [&](const SetSelectedFacesSmooth &a) {
                for_each_edit_selection(Element::Face, [&](entt::entity me, const Mesh &mesh, const uint32_t *bits, MeshSelectionBitsetRange br) {
                    auto sharp = meshes.GetFaceSharpness(mesh.GetStoreId());
                    ::selection::ForEachSelected(bits, br.Offset, br.Count, [&](uint32_t f) {
                        if (f < sharp.size()) sharp[f] = a.Smooth ? 0 : 1;
                    });
                    r.emplace_or_replace<MeshShadingDirty>(me);
                });
            },
            [&](const SetSelectedEdgesSharp &a) {
                for_each_edit_selection(Element::Edge, [&](entt::entity me, const Mesh &mesh, const uint32_t *bits, MeshSelectionBitsetRange br) {
                    auto sharp = meshes.GetEdgeSharpness(mesh.GetStoreId());
                    ::selection::ForEachSelected(bits, br.Offset, br.Count, [&](uint32_t e) {
                        if (e < sharp.size()) sharp[e] = a.Sharp ? 1 : 0;
                    });
                    r.emplace_or_replace<MeshShadingDirty>(me);
                });
            },
            [&](const SetSelectedVertexEdgesSharp &a) {
                for_each_edit_selection(Element::Vertex, [&](entt::entity me, const Mesh &mesh, const uint32_t *bits, MeshSelectionBitsetRange br) {
                    auto sharp = meshes.GetEdgeSharpness(mesh.GetStoreId());
                    ::selection::ForEachVertexTouchedEdge(bits, br.Offset, br.Count, mesh, [&](uint32_t e) {
                        if (e < sharp.size()) sharp[e] = a.Sharp ? 1 : 0;
                    });
                    r.emplace_or_replace<MeshShadingDirty>(me);
                });
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
                const auto [mesh_entity, _] = ::AddMesh(r, meshes, meshes.CreateMesh(primitive::CreateMesh(a.Shape), {}, {}, true), *a.Info);
                r.emplace<PrimitiveShape>(mesh_entity, a.Shape);
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
                    else return glm::clamp(v, a.Min, a.Max);
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
                    if (a.Mask != 0u) r.emplace_or_replace<PbrMeshFeatures>(e, a.Mask);
                    else r.remove<PbrMeshFeatures>(e);
                });
            },
            [&]<typename Field>(const Update<Field> &a) { ApplyUpdate(r, viewport, a); },
            // Mesh-data components (material assignment / slot selection) live on the object's mesh entity.
            [&]<typename T>(const Replace<T> &a) { for_each_mesh_target(a.Scope, a.Entity, [&](entt::entity e) { r.emplace_or_replace<T>(e, a.Value); }); },
            [&](const Replace<PunctualLight> &a) {
                auto set_light = [&](entt::entity e) {
                    const auto *old = r.try_get<const PunctualLight>(e);
                    const auto &n = a.Value;
                    if (!old || old->Type != n.Type || old->Range != n.Range || old->OuterConeCos != n.OuterConeCos || old->InnerConeCos != n.InnerConeCos) {
                        r.emplace_or_replace<LightWireframeDirty>(e);
                    }
                    r.emplace_or_replace<PunctualLight>(e, n);
                };
                ForEachReplaceTarget<PunctualLight>(r, a.Scope, a.Entity, set_light);
            },
        },
        action
    );
}
} // namespace action::object
