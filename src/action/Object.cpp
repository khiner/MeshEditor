#include "action/Object.h"
#include "Camera.h"
#include "Profile.h"
#include "action/Dispatch.h"
#include "action/Errors.h"
#include "animation/MorphWeights.h"
#include "armature/Armature.h"
#include "mesh/MeshClone.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshStores.h"
#include "mesh/Primitives.h"
#include "object/ObjectOps.h"
#include "procedure/Procedure.h"
#include "render/GpuBuffers.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "render/MeshletBuildGpu.h"
#include "render/RenderChanges.h"
#include "scene/CameraLens.h"
#include "scene/Defaults.h"
#include "scene/SceneGraphOps.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "selection/SelectionState.h"
#include "viewport/InteractionComponents.h"

#include <format>

namespace {
// Create an armature object over `data_entity`, creating fresh armature data when null.
state::Entity CreateArmatureObject(state::Registry &r, MeshStore &meshes, state::Entity data_entity, std::string_view name, const Transform &transform, MeshInstanceCreateInfo::SelectBehavior select) {
    if (data_entity == state::Null) {
        data_entity = r.create();
        r.emplace<Armature>(data_entity);
    }
    const auto entity = r.create();
    EmplaceObject(r, entity, ObjectType::Armature);
    r.emplace<ArmatureObject>(entity, data_entity);
    r.emplace<Transform>(entity, transform);
    EmplaceUniqueName(r, entity, name.empty() ? "Armature" : name);
    ::ApplySelectBehavior(r, entity, select);
    ::CreateBoneInstances(r, meshes, entity, data_entity);
    return entity;
}

// A mesh instance's duplicate takes `clone`, its mesh's cloned store record.
// The first duplicate of a clone creates its mesh entity, and the duplicates after it instance that mesh.
state::Entity DuplicateOne(state::Registry &r, state::Entity e, uint32_t clone) {
    auto &meshes = r.Resources.get<MeshStore>();
    const ObjectCreateInfo create_info{
        .Name = std::format("{}_copy", GetName(r, e)),
        // Duplicate is created at root, so its local must match source's world.
        .Transform = *WorldTransformOf(r, e),
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

    const MeshInstanceCreateInfo info{.Name = create_info.Name, .Transform = create_info.Transform, .Select = create_info.Select, .Visible = !r.all_of<Hidden>(e)};
    const auto instance = [&] {
        if (const auto clone_mesh = MeshEntityOf(r, clone); clone_mesh != state::Null) return ::AddMeshInstance(r, clone_mesh, info);
        const auto [clone_mesh, clone_instance] = ::AddMesh(r, clone, info);
        const auto mesh_entity = r.get<const Instance>(e).Entity;
        if (const auto *prim_shape = r.try_get<const PrimitiveShape>(mesh_entity)) r.emplace<PrimitiveShape>(clone_mesh, *prim_shape);
        if (r.all_of<procedure::GeometryProcedure>(mesh_entity)) action::nodes::CopyProcedure(r, mesh_entity, clone_mesh);
        return clone_instance;
    }();
    if (const auto *armature_modifier = r.try_get<ArmatureModifier>(e)) r.emplace<ArmatureModifier>(instance, *armature_modifier);
    if (const auto *bone_attachment = r.try_get<BoneAttachment>(e)) r.emplace<BoneAttachment>(instance, *bone_attachment);
    if (const auto *weights = r.try_get<const MorphWeightRange>(e)) r.emplace<MorphWeightRange>(instance, r.Resources.get<GpuBuffers>().MorphWeightBuffer.Clone(weights->Weights));
    return instance;
}

state::Entity DuplicateLinkedOne(state::Registry &r, state::Entity e) {
    auto &meshes = r.Resources.get<MeshStore>();
    if (r.all_of<BoneSubPartOf>(e)) return state::Null;
    if (!r.all_of<Instance>(e)) {
        const auto select_behavior = r.all_of<Selected>(e) ? MeshInstanceCreateInfo::SelectBehavior::Additive : MeshInstanceCreateInfo::SelectBehavior::None;

        if (const auto *armature = r.try_get<ArmatureObject>(e)) {
            return CreateArmatureObject(r, meshes, armature->Entity, std::format("{}_copy", GetName(r, e)), *WorldTransformOf(r, e), select_behavior);
        }
        return ::AddEmpty(r, meshes, {.Name = std::format("{}_copy", GetName(r, e)), .Transform = *WorldTransformOf(r, e), .Select = select_behavior});
    }

    const auto mesh_entity = r.get<Instance>(e).Entity;
    const auto e_new = r.create();
    EmplaceUniqueName(r, e_new, NameStem(GetName(r, e)));
    r.emplace<Instance>(e_new, mesh_entity);
    EmplaceObject(r, e_new, ObjectType::Mesh);
    const Transform t_new{*WorldTransformOf(r, e)};
    r.emplace_or_replace<Transform>(e_new, t_new);
    if (const auto *armature_modifier = r.try_get<ArmatureModifier>(e)) r.emplace<ArmatureModifier>(e_new, *armature_modifier);
    if (const auto *bone_attachment = r.try_get<BoneAttachment>(e)) r.emplace<BoneAttachment>(e_new, *bone_attachment);
    if (const auto *weights = r.try_get<const MorphWeightRange>(e)) r.emplace<MorphWeightRange>(e_new, r.Resources.get<GpuBuffers>().MorphWeightBuffer.Clone(weights->Weights));

    r.emplace<Selected>(e_new);

    return e_new;
}
} // namespace

namespace action {
bool UpdateTraits<PrimitiveShape>::Has(const state::Registry &r, state::Entity e) { return r.all_of<PrimitiveShape>(e); }
state::Entity UpdateTraits<PrimitiveShape>::Active(const state::Registry &r) {
    const auto e = GetActiveMeshEntity(r);
    return e != state::Null && Has(r, e) ? e : state::Null;
}
// Only meshes of the active primitive's kind share its fields.
void UpdateTraits<PrimitiveShape>::ForEachSelected(state::Registry &r, const std::function<void(state::Entity)> &fn) {
    const auto active = Active(r);
    if (active == state::Null) return;
    const auto kind = r.get<const PrimitiveShape>(active).index();
    for (const auto e : r.get<const SelectionFlags>(state::Document).Meshes) {
        if (Has(r, e) && r.get<const PrimitiveShape>(e).index() == kind) fn(e);
    }
}
void UpdateTraits<PrimitiveShape>::Read(const state::Registry &r, state::Entity e, uint16_t offset, void *dst, size_t size) {
    std::visit([&](const auto &alt) { std::memcpy(dst, reinterpret_cast<const std::byte *>(&alt) + offset, size); }, r.get<const PrimitiveShape>(e));
}
void UpdateTraits<PrimitiveShape>::Write(state::Registry &r, state::Entity e, uint16_t offset, const void *src, size_t size) {
    if (::selection::HasScaleLockedInstance(r, e)) return;
    r.patch<PrimitiveShape>(e, [&](PrimitiveShape &s) { std::visit([&](auto &alt) { std::memcpy(reinterpret_cast<std::byte *>(&alt) + offset, src, size); }, s); });
}
const FieldSpec &UpdateTraits<PrimitiveShape>::Bounds(const state::Registry &r, state::Entity e, uint16_t offset) {
    return std::visit([&](const auto &alt) -> const FieldSpec & { return FieldAt<std::remove_cvref_t<decltype(alt)>>(offset).Spec; }, r.get<const PrimitiveShape>(e));
}
} // namespace action

namespace action::object {
void Apply(state::Registry &r, const Action &action) {
    auto &meshes = r.Resources.get<MeshStore>();
    auto begin_translate = [&] { r.emplace_or_replace<StartScreenTransform>(state::Document, TransformGizmo::TransformType::Translate); };
    const auto duplicate = [&](bool linked) {
        if (!(linked ? CanDuplicateLinked(r) : CanDuplicate(r))) return;
        const profile::CpuScope scope{linked ? "DuplicateLinked" : "Duplicate"};
        const auto entities = SelectedIn(r);
        // Every duplicated mesh clones once in one batch, with its arenas reserved once, and instances sharing a mesh share its clone.
        std::vector<uint32_t> sources, source_of(entities.size(), InvalidOffset);
        if (!linked) {
            std::unordered_map<state::Entity, uint32_t> mesh_sources;
            for (uint32_t i = 0u; i < entities.size(); ++i) {
                const auto e = entities[i];
                if (!r.all_of<Instance>(e) || r.all_of<BoneSubPartOf>(e)) continue;
                const auto mesh_entity = r.get<Instance>(e).Entity;
                if (r.all_of<ObjectExtrasTag>(mesh_entity) || !HasMesh(r, mesh_entity)) continue;
                const auto [source, added] = mesh_sources.try_emplace(mesh_entity, uint32_t(sources.size()));
                if (added) {
                    meshes.PlanClone(GetMesh(r, mesh_entity));
                    sources.push_back(r.get<const MeshHandle>(mesh_entity).StoreId);
                }
                source_of[i] = source->second;
            }
            meshes.CommitReserves();
        }
        // The clones' store records and render records copy together, in one submit after their entities exist.
        CloneCopies copies;
        const auto clones = meshes.CloneMeshes(copies, sources, true);
        for (uint32_t i = 0u; i < entities.size(); ++i) {
            const auto src = entities[i];
            const auto dup = linked ? DuplicateLinkedOne(r, src) : DuplicateOne(r, src, source_of[i] == InvalidOffset ? InvalidOffset : clones[source_of[i]]);
            if (r.all_of<Active>(src)) {
                r.remove<Active>(src);
                r.emplace<Active>(dup);
            }
            r.remove<Selected>(src);
        }
        if (!clones.empty()) {
            mtl::ComputeChain chain{meshes.BufferContext()};
            copies.Encode(chain, GetMeshPipelines(r));
            chain.Submit();
            meshes.RebuildSelectionIndex(clones);
            // A clone with render records draws them at once and takes its source's pending repairs.
            for (const auto id : clones) {
                RefreshMeshBinding(r, id);
                const auto &record = meshes.Get(id);
                const auto entity = MeshEntityOf(r, id);
                if (entity == state::Null || record.RenderTopologies == 0u) continue;
                if (meshes.HasDirtyPositions(record)) r.emplace_or_replace<MeshletPositionsDirty>(entity);
                if (meshes.HasDirtyGroups(record)) r.emplace_or_replace<ClusterLodDirty>(entity);
                if (!meshes.ClusterGroupCount(record)) r.emplace_or_replace<ClusterLodDemand>(entity);
            }
        }
        begin_translate();
    };
    // Mesh-data components live on the object's mesh entity.
    auto for_each_mesh_target = [&](const Target &target, auto &&fn) {
        ForEachTarget(
            target,
            [&] { return GetActiveMeshEntity(r); },
            [&](auto &&f) { for (const auto e : r.get<const SelectionFlags>(state::Document).Meshes) f(e); },
            fn
        );
    };
    const auto edit_selection_meshes = [&](Element element) {
        std::vector<state::Entity> result;
        if (r.get<const EditMode>(state::Document).Value != element) return result;
        for (const auto me : r.view<const MeshElementSelection>()) {
            if (!HasMesh(r, me)) continue;
            const auto &summary = meshes.GetSelectionSummary(*DrawnStoreId(r, me));
            if (summary.Mode == element && summary.SelectedCount > 0) result.push_back(me);
        }
        return result;
    };
    // Append a node with the parameters to each procedure mesh and return the other meshes, which edit in place.
    const auto append_to_procedures = [&]<typename N>(std::span<const state::Entity> mesh_entities, const N &parameters) {
        std::vector<state::Entity> targets;
        for (const auto e : mesh_entities) {
            if (!r.all_of<procedure::GeometryProcedure>(e)) {
                targets.push_back(e);
                continue;
            }
            const auto node = nodes::AppendNode(r, e, procedure::KindOf<N>);
            if (node == state::Null) break;
            r.replace<N>(node, parameters);
        }
        return targets;
    };
    std::visit(
        overloaded{
            [&](Delete) {
                if (!CanDelete(r)) return;
                Destroy(r, SelectedIn(r) | std::views::filter([&](state::Entity e) { return !r.all_of<SubElementOf>(e); }) | std::ranges::to<std::vector>());
            },
            [&](Duplicate) { duplicate(false); },
            [&](DuplicateLinked) { duplicate(true); },
            [&](ToggleHidden) {
                for (const auto e : SelectedIn(r)) {
                    if (r.all_of<Instance>(e) && !r.all_of<Hidden>(e)) Hide(r, e);
                    else Show(r, e);
                }
            },
            [&](const SetSelectedVisible &a) {
                for (const auto e : SelectedIn(r)) {
                    if (!r.all_of<Instance>(e) || r.all_of<SubElementOf>(e)) continue;
                    if (a.Visible) Show(r, e);
                    else Hide(r, e);
                }
            },
            [&](const SetSelectedSmoothShading &a) {
                const auto targets = append_to_procedures(r.get<const SelectionFlags>(state::Document).Meshes, procedure::ShadeSmooth{a.Smooth});
                ApplyEditSharpness(r, targets, a.Smooth ? EditSharpnessOperation::SmoothAll : EditSharpnessOperation::SetAllFaces, !a.Smooth);
            },
            [&](const ShadeSelectedSmoothByAngle &a) {
                const auto targets = append_to_procedures(r.get<const SelectionFlags>(state::Document).Meshes, procedure::ShadeSmoothByAngle{a.Angle});
                ApplyEditSharpness(r, targets, EditSharpnessOperation::SmoothByAngle, false, a.Angle);
            },
            [&](const SetSelectedSharp &a) {
                const auto targets = append_to_procedures(edit_selection_meshes(a.Element), procedure::MarkSharp{a.Sharp});
                const auto operation = a.Element == Element::Face ? EditSharpnessOperation::SetSelectedFaces :
                    a.Element == Element::Edge                    ? EditSharpnessOperation::SetSelectedEdges :
                                                                    EditSharpnessOperation::SetVertexEdges;
                ApplyEditSharpness(r, targets, operation, a.Sharp);
            },
            [&](ParentToActive) {
                const auto active = FindActiveEntity(r);
                if (active == state::Null) return;
                if (SetParentKeepWorld(r, SelectedIn(r), active)) action::Fail(r, "Loop in parents");
            },
            [&](ClearParent) {
                ::ClearParents(r, SelectedIn(r));
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
            [&](const ImportMesh &a) { RequestImportMesh(r, a.Path, *a.Info); },
            [&](const SetPbrMeshFeaturesMask &a) {
                for_each_mesh_target(a.Target, [&](state::Entity e) {
                    if (a.Mask != 0u) r.emplace_or_replace<PbrMeshFeatures>(e, a.Mask);
                    else r.remove<PbrMeshFeatures>(e);
                });
            },
            [&]<typename T>(const UpdateMaterial<T> &a) {
                auto &materials = r.Resources.get<GpuBuffers>().Materials;
                if (a.Index >= materials.Count<PBRMaterial>()) return;
                materials.Update(std::as_bytes(std::span{&a.Value, 1}), uint64_t(a.Index) * sizeof(PBRMaterial) + a.Offset);
                r.Resources.get<RenderChanges>().Materials.emplace(state::Document);
            },
            [&](const SetMaterialSlotSelection &a) {
                for_each_mesh_target(a.Target, [&](state::Entity e) { r.emplace_or_replace<MeshMaterialSlotSelection>(e, a.PrimitiveIndex); });
            },
            [&](const SetMaterialAssignment &a) {
                for_each_mesh_target(a.Target, [&](state::Entity e) { r.emplace_or_replace<MeshMaterialAssignment>(e, a.PrimitiveIndex, a.MaterialIndex); });
            },
            [&](const SetProjection &a) {
                ForEachTarget(
                    a.Target,
                    [&] { const auto e = FindActiveEntity(r); return HasLens(r, e) ? e : state::Null; },
                    [&](auto &&fn) {
                        for (const auto e : SelectedIn(r)) {
                            if (HasLens(r, e)) fn(e);
                        }
                    },
                    [&](state::Entity e) {
                        const float distance = std::max(Length(WorldTransformOf(r, e)->P), 1.f);
                        if (const auto *perspective = r.try_get<const Perspective>(e); perspective && a.Orthographic) SetLens(r, e, OrthographicFromPerspective(*perspective, distance));
                        else if (const auto *orthographic = r.try_get<const Orthographic>(e); orthographic && !a.Orthographic) SetLens(r, e, PerspectiveFromOrthographic(*orthographic, distance));
                    }
                );
            },
            [&](const SetLightType &a) {
                ForEachComponentTarget<PunctualLight>(r, a.Target, [&](auto e) {
                    r.patch<PunctualLight>(e, [&](auto &light) {
                        auto next = Defaults::MakePunctualLight(a.Type);
                        next.Color = light.Color;
                        next.Intensity = light.Intensity;
                        light = next;
                    });
                });
            },
            [&](const SetSpotCone &a) {
                ForEachComponentTarget<PunctualLight>(r, a.Target, [&](auto e) {
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
