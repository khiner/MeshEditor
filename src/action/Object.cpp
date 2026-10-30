#include "action/Object.h"
#include "Camera.h"
#include "Profile.h"
#include "SortUnique.h"
#include "action/Dispatch.h"
#include "action/Errors.h"
#include "animation/MorphWeights.h"
#include "armature/Armature.h"
#include "mesh/MeshClone.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshStores.h"
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
#include "selection/SelectionState.h"
#include "viewport/InteractionComponents.h"

#include <format>

namespace {
// Create an armature object over `data_entity`, creating fresh armature data when null.
state::Entity CreateArmatureObject(state::Registry &r, MeshStore &meshes, state::Entity data_entity, std::string_view name, const Transform &transform, MeshInstanceCreateInfo::SelectBehavior select, state::Entity scene = state::Null) {
    if (data_entity == state::Null) {
        data_entity = r.create();
        r.emplace<Armature>(data_entity);
    }
    const auto entity = r.create();
    EmplaceObject(r, entity, ObjectType::Armature, scene);
    r.emplace<ArmatureObject>(entity, data_entity);
    r.emplace<Transform>(entity, transform);
    EmplaceUniqueName(r, entity, name.empty() ? "Armature" : name);
    ::ApplySelectBehavior(r, entity, select);
    ::CreateBoneInstances(r, meshes, entity, data_entity);
    return entity;
}

// A duplicated mesh entity, the index of its drawn record's clone, and its copy once the first duplicate creates it.
struct MeshCopy {
    state::Entity Mesh;
    uint32_t Clone;
    state::Entity Copy{state::Null};
};

// A mesh instance's duplicate is an instance of its mesh's copy, which the first duplicate creates from the mesh's cloned record.
// A procedure mesh's copy holds a copy of its procedure and draws the clone, and an owned mesh's copy is a mesh whose Source node holds the clone.
state::Entity DuplicateOne(state::Registry &r, state::Entity e, MeshCopy *copy, std::span<const uint32_t> clones) {
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
        return ::AddEmpty(r, create_info);
    }

    // Bone sub-entities (head/tail joints, bone instances) are not independently duplicable.
    if (r.all_of<BoneSubPartOf>(e)) return state::Null;

    // Object extras (Camera, Empty, Light) have Instance but create their own wireframe mesh.
    if (r.all_of<ObjectExtrasTag>(r.get<Instance>(e).Entity)) {
        if (const auto lens = LensOf(r, e)) return ::AddCamera(r, create_info, *lens);
        if (const auto *light = r.try_get<const PunctualLight>(e)) return ::AddLight(r, create_info, *light);
        return ::AddEmpty(r, create_info);
    }

    const MeshInstanceCreateInfo info{.Name = create_info.Name, .Transform = create_info.Transform, .Select = create_info.Select, .Visible = !r.all_of<Hidden>(e)};
    if (copy->Copy == state::Null) {
        const auto clone = clones[copy->Clone];
        copy->Copy = HasMesh(r, copy->Mesh) ? action::nodes::CopyProcedure(r, copy->Mesh, clone) : ::AddMeshData(r, clone);
    }
    const auto instance = ::AddMeshInstance(r, copy->Copy, info);
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
        return ::AddEmpty(r, {.Name = std::format("{}_copy", GetName(r, e)), .Transform = *WorldTransformOf(r, e), .Select = select_behavior});
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

namespace action::object {
void Apply(state::Registry &r, const Action &action) {
    auto &meshes = r.Resources.get<MeshStore>();
    auto begin_translate = [&] { r.emplace_or_replace<StartScreenTransform>(state::Document, TransformGizmo::TransformType::Translate); };
    const auto duplicate = [&](bool linked) {
        if (!(linked ? CanDuplicateLinked(r) : CanDuplicate(r))) return;
        const profile::CpuScope scope{linked ? "DuplicateLinked" : "Duplicate"};
        auto entities = SelectedIn(r);
        const auto selected = entities;
        std::erase_if(entities, [&](state::Entity e) {
            for (auto parent = ParentOrNull(r, e); parent != state::Null; parent = ParentOrNull(r, parent))
                if (std::ranges::contains(selected, parent)) return true;
            return false;
        });
        const auto hierarchy = [&](state::Entity e) {
            const auto *object = r.try_get<const Instance>(e);
            const auto *generated = object ? r.try_get<const procedure::ProcedureInstances>(object->Entity) : nullptr;
            const bool children = std::ranges::any_of(Children{&r, e}, [&](state::Entity child) { return r.all_of<ObjectKind>(child) && (!generated || !std::ranges::contains(generated->Entities, child)); });
            if (linked && r.all_of<ArmatureObject>(e) && !children && !r.all_of<HierarchyInstance>(e)) return false;
            return r.all_of<HierarchyInstance>(e) || children || (r.all_of<ObjectKind>(e) && r.get<const ObjectKind>(e).Value != ObjectType::Mesh);
        };
        // Every duplicated mesh's drawn record clones once in one batch, with its arenas reserved once, and instances sharing a mesh share its copy.
        std::vector<uint32_t> sources, copy_of(entities.size(), InvalidOffset);
        std::vector<MeshCopy> copies_of;
        if (!linked) {
            std::unordered_map<state::Entity, uint32_t> mesh_copies;
            for (uint32_t i = 0u; i < entities.size(); ++i) {
                const auto e = entities[i];
                if (hierarchy(e) || !r.all_of<Instance>(e) || r.all_of<BoneSubPartOf>(e)) continue;
                const auto mesh_entity = r.get<Instance>(e).Entity;
                if (r.all_of<ObjectExtrasTag>(mesh_entity) || !DrawsMesh(r, mesh_entity)) continue;
                const auto [copy, added] = mesh_copies.try_emplace(mesh_entity, uint32_t(copies_of.size()));
                if (added) {
                    const auto id = *DrawnStoreId(r, mesh_entity);
                    copies_of.push_back({mesh_entity, uint32_t(sources.size())});
                    meshes.PlanClone(Mesh{meshes, id});
                    sources.push_back(id);
                }
                copy_of[i] = copy->second;
            }
            meshes.CommitReserves();
        }
        // The clones' store records and render records copy together, in one submit after their entities exist.
        CloneCopies copies;
        const auto clones = meshes.CloneMeshes(copies, sources, true);
        for (uint32_t i = 0u; i < entities.size(); ++i) {
            const auto src = entities[i];
            const auto dup = hierarchy(src) ? CreateHierarchyInstance(r, src, *WorldTransformOf(r, src)) : linked ? DuplicateLinkedOne(r, src) :
                                                                                                                    DuplicateOne(r, src, copy_of[i] == InvalidOffset ? nullptr : &copies_of[copy_of[i]], clones);
            if (hierarchy(src)) {
                if (!linked) MakeHierarchyIndependent(r, dup);
                r.emplace_or_replace<Selected>(dup);
            }
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
    // The edit meshes with a selection in the element domain, which is the edit domain or none.
    const auto edit_selection_meshes = [&](Element element) {
        return r.get<const EditMode>(state::Document).Value == element ? mesh::SelectedEditMeshes(r) : std::vector<state::Entity>{};
    };
    // Append a node with the parameters to each mesh.
    const auto append = [&]<typename N>(std::span<const state::Entity> mesh_entities, const N &parameters) {
        for (const auto e : mesh_entities) {
            const auto node = nodes::AppendNode(r, e, procedure::KindOf<N>);
            if (node == state::Null) break;
            r.replace<N>(node, parameters);
        }
    };
    const bool valid_creation = std::visit([&](const auto &a) {
        if constexpr (requires { a.Info; }) {
            if (!a.Info || (a.Info->Scene != state::Null && !r.all_of<Scene>(a.Info->Scene))) return false;
            return a.Info->Network == state::Null || (r.all_of<procedure::NetworkSubnet>(a.Info->Network) && procedure::NetworkRoot(r, a.Info->Network) == (a.Info->Scene == state::Null ? FrameScene(r) : a.Info->Scene));
        } else return true;
    },
                                           action);
    if (!valid_creation) return action::Fail(r, "The creation scene does not exist");
    std::visit(
        overloaded{
            [&](Delete) {
                if (!CanDelete(r)) return;
                auto removed = SelectedIn(r) | std::views::filter([&](state::Entity e) { return !r.all_of<SubElementOf>(e); }) | std::views::transform([&](state::Entity e) { return HierarchyEditTarget(r, e, true); }) | std::ranges::to<std::vector>();
                SortUnique(removed);
                Destroy(r, removed);
            },
            [&](Duplicate) { duplicate(false); },
            [&](DuplicateLinked) { duplicate(true); },
            [&](const InstantiateHierarchy &a) {
                if (!r.all_of<ObjectKind>(a.Source)) return Fail(r, "Choose an object hierarchy");
                const auto root = CreateHierarchyInstance(r, a.Source, ComposeTransform(r, a.Source, state::Null, {}));
                if (root != state::Null) ApplySelectBehavior(r, root, MeshInstanceCreateInfo::SelectBehavior::Exclusive);
            },
            [&](const MakeIndependent &a) { MakeHierarchyIndependent(r, a.Root); },
            [&](ToggleHidden) {
                for (const auto e : SelectedIn(r)) {
                    const auto target = HierarchyEditTarget(r, e, true);
                    if (!r.all_of<Hidden>(e)) Hide(r, target);
                    else Show(r, target);
                }
            },
            [&](const SetVisible &a) {
                ForEachTarget(a.Target, [&] { return FindActiveEntity(r); }, [&](auto &&fn) { for (const auto e : SelectedIn(r)) fn(e); }, [&](state::Entity e) {
                    if (!r.all_of<ObjectKind>(e) || r.all_of<SubElementOf>(e)) return;
                    e = HierarchyEditTarget(r, e, true);
                    if (a.Visible) Show(r, e);
                    else Hide(r, e); });
            },
            [&](const SetSelectedSmoothShading &a) { append(r.get<const SelectionFlags>(state::Document).Meshes, procedure::ShadeSmooth{a.Smooth}); },
            [&](const ShadeSelectedSmoothByAngle &a) { append(r.get<const SelectionFlags>(state::Document).Meshes, procedure::ShadeSmoothByAngle{a.Angle}); },
            [&](const SetSelectedSharp &a) { append(edit_selection_meshes(a.Element), procedure::MarkSharp{a.Sharp}); },
            [&](ParentToActive) {
                const auto active = FindActiveEntity(r);
                if (active == state::Null) return;
                std::vector<std::pair<state::Entity, state::Entity>> targets;
                for (const auto e : SelectedIn(r))
                    if (e != active) targets.push_back(HierarchyReferenceTarget(r, e, active, true));
                SortUnique(targets);
                for (const auto [entity, parent] : targets)
                    if (HierarchyParentCycle(r, entity, parent)) return Fail(r, "Loop in parents");
                const auto owned = r.view<const BodyPoseCache>() | std::ranges::to<std::vector>();
                for (const auto [entity, parent] : targets) SetParentKeepWorld(r, std::array{entity}, parent, owned);
            },
            [&](const ClearParent &a) {
                std::vector<state::Entity> targets;
                ForEachTarget(a.Target, [&] { return FindActiveEntity(r); }, [&](auto &&fn) { for (const auto e : SelectedIn(r)) fn(e); }, [&](state::Entity e) {
                    if (r.valid(e)) targets.push_back(HierarchyEditTarget(r, e, true)); });
                SortUnique(targets);
                ::ClearParents(r, targets);
            },
            [&](const SetParent &a) {
                if (!r.all_of<ObjectKind>(a.Entity) || !r.all_of<ObjectKind>(a.Parent) || r.any_of<SubElementOf, BoneIndex>(a.Entity) || r.any_of<SubElementOf, BoneIndex>(a.Parent)) return action::Fail(r, "Parenting requires two scene objects");
                const auto [entity, parent] = HierarchyReferenceTarget(r, a.Entity, a.Parent, true);
                if (HierarchyParentCycle(r, entity, parent)) return Fail(r, "Loop in parents");
                SetParentKeepWorld(r, std::array{entity}, parent, r.view<const BodyPoseCache>() | std::ranges::to<std::vector>());
            },
            [&](const Rename &a) {
                if (!r.all_of<ObjectKind>(a.Entity) || r.any_of<SubElementOf, BoneIndex>(a.Entity)) return action::Fail(r, "Renaming requires a scene object");
                const auto entity = HierarchyEditTarget(r, a.Entity, true);
                r.remove<Name>(entity);
                EmplaceUniqueName(r, entity, a.Name);
            },
            [&](const AddEmpty &a) { ::AddEmpty(r, *a.Info); if (a.Info->BeginTranslate) begin_translate(); },
            [&](const AddArmature &a) {
                const auto entity = CreateArmatureObject(r, meshes, state::Null, a.Info->Name, a.Info->Transform, a.Info->Select, a.Info->Scene);
                procedure::PlaceInNetwork(r, a.Info->Network, entity);
                if (a.Info->BeginTranslate) begin_translate();
            },
            [&](const AddCamera &a) { ::AddCamera(r, *a.Info, a.Props); if (a.Info->BeginTranslate) begin_translate(); },
            [&](const AddLight &a) { ::AddLight(r, *a.Info); if (a.Info->BeginTranslate) begin_translate(); },
            [&](const AddMeshPrimitive &a) {
                ::AddMeshPrimitive(r, a.Parameters, *a.Info);
                if (a.Info->BeginTranslate) begin_translate();
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
