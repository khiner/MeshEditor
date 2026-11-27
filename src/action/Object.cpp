#include "action/Object.h"
#include "Camera.h"
#include "File.h"
#include "Profile.h"
#include "SortUnique.h"
#include "action/Dispatch.h"
#include "action/Errors.h"
#include "animation/Fields.h"
#include "animation/MorphWeights.h"
#include "armature/Armature.h"
#include "gltf/SourceAssets.h"
#include "graph/Graph.h"
#include "mesh/MeshClone.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshStores.h"
#include "object/ObjectOps.h"
#include "physics/PhysicsTypes.h"
#include "project/Assets.h"
#include "render/GpuBufferOps.h"
#include "render/GpuBuffers.h"
#include "render/GpuSceneState.h"
#include "render/MaterialComponents.h"
#include "render/MeshletBuildGpu.h"
#include "render/ObjectGeometry.h"
#include "render/RenderChanges.h"
#include "render/Textures.h"
#include "scene/CameraLens.h"
#include "scene/Defaults.h"
#include "scene/SceneGraphOps.h"
#include "selection/SelectionState.h"
#include "viewport/InteractionComponents.h"

#include <format>

namespace {
// Create an armature object over `data_entity`, creating fresh armature data when null.
state::Entity CreateArmatureObject(state::Registry &r, MeshStore &meshes, state::Entity data_entity, std::string_view name, const Transform &transform, MeshObjectCreateInfo::SelectBehavior select, state::Entity scene = state::Null) {
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
    ::CreateBoneEntities(r, meshes, entity, data_entity);
    return entity;
}

// A duplicated mesh entity, the index of its drawn record's clone, and its copy once the first duplicate creates it.
struct MeshCopy {
    state::Entity Mesh;
    uint32_t Clone;
    state::Entity Copy{state::Null};
};

// A graph mesh's copy holds a copy of its graph and draws the clone, and an owned mesh's copy is a mesh whose Source node holds the clone.
state::Entity DuplicateOne(state::Registry &r, state::Entity e, MeshCopy *copy, std::span<const uint32_t> clones) {
    auto &meshes = r.Resources.get<MeshStore>();
    const ObjectCreateInfo create_info{
        .Name = std::string{NameStem(GetName(r, e))},
        .Transform = *WorldTransformOf(r, e),
        .Select = r.all_of<Selected>(e) ? MeshObjectCreateInfo::SelectBehavior::Additive : MeshObjectCreateInfo::SelectBehavior::None,
    };

    if (!r.all_of<ObjectGeometry>(e)) {
        if (const auto object_type = r.all_of<ObjectKind>(e) ? r.get<const ObjectKind>(e).Value : ObjectType::Empty; object_type == ObjectType::Armature) {
            const auto copy_entity = CreateArmatureObject(r, meshes, state::Null, create_info.Name, create_info.Transform, create_info.Select);
            if (const auto *src_armature = r.try_get<ArmatureObject>(e)) {
                r.edit<Armature>(r.get<const ArmatureObject>(copy_entity).Entity) = r.get<const Armature>(src_armature->Entity);
            }
            return copy_entity;
        }
        return ::AddEmpty(r, create_info);
    }

    // Bone sub-entities, the head and tail joints and bone entities, duplicate only with their bone.

    // Camera, Empty and Light objects have an ObjectGeometry but create their own wireframe mesh.
    if (r.all_of<ObjectExtrasTag>(r.get<ObjectGeometry>(e).Entity)) {
        if (const auto lens = LensOf(r, e)) return ::AddCamera(r, create_info, *lens);
        if (const auto *light = r.try_get<const SceneLight>(e)) return ::AddLight(r, create_info, *light);
        return ::AddEmpty(r, create_info);
    }

    const MeshObjectCreateInfo info{.Name = create_info.Name, .Transform = create_info.Transform, .Select = create_info.Select, .Visible = !r.all_of<Hidden>(e)};
    if (copy->Copy == state::Null) {
        const auto clone = clones[copy->Clone];
        copy->Copy = HasMesh(r, copy->Mesh) ? action::nodes::CopyGraph(r, copy->Mesh, clone) : ::AddMeshData(r, clone);
    }
    const auto duplicate = ::AddMeshObject(r, copy->Copy, info);
    if (const auto *armature_modifier = r.try_get<ArmatureModifier>(e)) r.emplace<ArmatureModifier>(duplicate, *armature_modifier);
    if (const auto *bone_attachment = r.try_get<BoneAttachment>(e)) r.emplace<BoneAttachment>(duplicate, *bone_attachment);
    if (const auto *weights = r.try_get<const MorphWeightRange>(e)) r.emplace<MorphWeightRange>(duplicate, r.Resources.get<GpuBuffers>().MorphWeightBuffer.Clone(weights->Weights));
    return duplicate;
}

// Copies the clones' store records and render records together, in one submit after their entities exist.
void FinishClones(state::Registry &r, CloneCopies &copies, std::span<const uint32_t> clones) {
    if (clones.empty()) return;
    auto &meshes = r.Resources.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    copies.Encode(chain, GetMeshPipelines(r));
    chain.Submit();
    meshes.RebuildSelectionIndex(clones);
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

state::Entity DuplicateLinkedOne(state::Registry &r, state::Entity e) {
    auto &meshes = r.Resources.get<MeshStore>();
    if (!r.all_of<ObjectGeometry>(e)) {
        const auto select_behavior = r.all_of<Selected>(e) ? MeshObjectCreateInfo::SelectBehavior::Additive : MeshObjectCreateInfo::SelectBehavior::None;

        if (const auto *armature = r.try_get<ArmatureObject>(e)) {
            return CreateArmatureObject(r, meshes, armature->Entity, NameStem(GetName(r, e)), *WorldTransformOf(r, e), select_behavior);
        }
        return ::AddEmpty(r, {.Name = std::string{NameStem(GetName(r, e))}, .Transform = *WorldTransformOf(r, e), .Select = select_behavior});
    }

    const auto mesh_entity = r.get<ObjectGeometry>(e).Entity;
    const auto e_new = r.create();
    EmplaceUniqueName(r, e_new, NameStem(GetName(r, e)));
    r.emplace<ObjectGeometry>(e_new, mesh_entity);
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
state::Entity CopyGeometry(state::Registry &r, state::Entity geometry) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto id = DrawnStoreId(r, geometry);
    if (!id) return geometry;
    meshes.PlanClone(Mesh{meshes, *id});
    meshes.CommitReserves();
    CloneCopies copies;
    const auto clones = meshes.CloneMeshes(copies, std::vector{*id}, true);
    const auto copy = HasMesh(r, geometry) ? nodes::CopyGraph(r, geometry, clones.front()) : ::AddMeshData(r, clones.front());
    FinishClones(r, copies, clones);
    return copy;
}
void Apply(state::Registry &r, const Action &action) {
    auto &meshes = r.Resources.get<MeshStore>();
    auto begin_translate = [&] { r.emplace_or_replace<StartScreenTransform>(state::Document, TransformGizmo::TransformType::Translate); };
    const auto duplicate = [&](bool linked) {
        if (!(linked ? CanDuplicateLinked(r) : CanDuplicate(r))) return;
        const profile::CpuScope scope{linked ? "DuplicateLinked" : "Duplicate"};
        auto entities = SelectedIn(r);
        // A part duplicates with its object.
        std::erase_if(entities, [&](state::Entity e) { return r.all_of<SubElementOf>(e); });
        if (entities.empty()) return;
        const auto has_instance = [&](state::Entity e) { return r.all_of<Instance>(e); };
        // A selected object below a selected parent duplicates in its parent's group, where its duplicate hangs below the parent's duplicate.
        // An object with an Instance heads no group, since it duplicates alone with an Instance of the same subgraph.
        const auto joins_parent = [&](state::Entity e) {
            const auto parent = ParentOrNull(r, e);
            return parent != state::Null && !has_instance(parent) && std::ranges::contains(entities, parent);
        };
        // Each group's root, then the selected objects below it.
        std::vector<std::vector<state::Entity>> groups;
        for (const auto e : entities)
            if (!joins_parent(e)) groups.push_back({e});
        for (const auto e : entities) {
            if (!joins_parent(e)) continue;
            auto root = e;
            while (joins_parent(root)) root = ParentOrNull(r, root);
            std::ranges::find(groups, root, [](const auto &group) { return group.front(); })->push_back(e);
        }
        // A group of several objects, or an object of a kind other than a mesh, duplicates through its copies.
        const auto copied = [&](const std::vector<state::Entity> &group) {
            const auto root = group.front();
            if (has_instance(root)) return false;
            if (group.size() > 1u) return true;
            if (linked && r.all_of<ArmatureObject>(root)) return false;
            return r.all_of<ObjectKind>(root) && r.get<const ObjectKind>(root).Value != ObjectType::Mesh;
        };
        std::vector<uint32_t> sources, copy_of(groups.size(), InvalidOffset);
        std::vector<MeshCopy> copies_of;
        if (!linked) {
            std::unordered_map<state::Entity, uint32_t> mesh_copies;
            for (uint32_t i = 0u; i < groups.size(); ++i) {
                const auto e = groups[i].front();
                if (has_instance(e) || copied(groups[i]) || !r.all_of<ObjectGeometry>(e)) continue;
                const auto mesh_entity = r.get<ObjectGeometry>(e).Entity;
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
        CloneCopies copies;
        const auto clones = meshes.CloneMeshes(copies, sources, true);
        // Each lone mesh object's duplicate, then the duplicates of the other groups together.
        std::vector<std::vector<state::Entity>> duplicates(groups.size()), together;
        for (uint32_t i = 0u; i < groups.size(); ++i) {
            const auto src = groups[i].front();
            if (has_instance(src)) {
                duplicates[i] = {AddInstance(r, r.get<const Instance>(src).SubgraphEntity, {.Name = std::string{NameStem(GetName(r, src))}, .Transform = *WorldTransformOf(r, src), .Select = MeshObjectCreateInfo::SelectBehavior::None}, r.get<const Instance>(src).PortValues)};
            } else if (copied(groups[i])) together.push_back(groups[i]);
            else duplicates[i] = {linked ? DuplicateLinkedOne(r, src) : DuplicateOne(r, src, copy_of[i] == InvalidOffset ? nullptr : &copies_of[copy_of[i]], clones)};
        }
        auto made = DuplicateObjects(r, together, linked);
        auto next = made.begin();
        for (uint32_t i = 0u; i < groups.size(); ++i) {
            const bool selects = has_instance(groups[i].front()) || copied(groups[i]);
            if (copied(groups[i])) duplicates[i] = std::move(*next++);
            for (size_t m = 0; m < groups[i].size(); ++m) {
                const auto src = groups[i][m], duplicate = duplicates[i][m];
                // A duplicate takes its source's place: below the source's parent at the source's local transform, and in the subgraph that holds the source.
                if (const auto parent = ParentOrNull(r, src); m == 0u && parent != state::Null) {
                    ::SetParent(r, duplicate, parent);
                    r.replace<Transform>(duplicate, r.get<const Transform>(src));
                }
                if (const auto holder = SubgraphHolding(r, src); holder != state::Null) graph::PlaceInGraph(r, holder, duplicate);
                if (selects) r.emplace_or_replace<Selected>(duplicate);
                if (r.all_of<Active>(src)) {
                    r.remove<Active>(src);
                    r.emplace<Active>(duplicate);
                }
                r.remove<Selected>(src);
            }
        }
        FinishClones(r, copies, clones);
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
        return r.get<const EditMode>(state::Document).Value == element ? mesh::SelectedEditMeshes(r) : std::vector<state::Entity>{};
    };
    const auto append = [&]<typename N>(std::span<const state::Entity> mesh_entities, const N &parameters) {
        for (const auto e : mesh_entities) {
            const auto node = nodes::AppendNode(r, e, graph::KindOf<N>);
            if (node == state::Null) break;
            r.replace<N>(node, parameters);
        }
    };
    const bool valid_creation = std::visit([&](const auto &a) {
        if constexpr (requires { a.Info; }) {
            if (!a.Info || (a.Info->Scene != state::Null && !r.all_of<Scene>(a.Info->Scene))) return false;
            return a.Info->Graph == state::Null || (r.all_of<graph::SubgraphItems>(a.Info->Graph) && graph::GraphRoot(r, a.Info->Graph) == (a.Info->Scene == state::Null ? FrameScene(r) : a.Info->Scene));
        } else return true;
    },
                                           action);
    if (!valid_creation) return action::Fail(r, "The creation scene does not exist");
    std::visit(
        overloaded{
            [&](Delete) {
                if (!CanDelete(r)) return;
                auto removed = SelectedIn(r) | std::views::filter([&](state::Entity e) { return !r.all_of<SubElementOf>(e); }) | std::views::transform([&](state::Entity e) { return SourceOfCopy(r, e); }) | std::ranges::to<std::vector>();
                SortUnique(removed);
                Destroy(r, removed);
            },
            [&](Duplicate) { duplicate(false); },
            [&](DuplicateLinked) { duplicate(true); },
            [&](const Instantiate &a) {
                if (!IsSceneSubgraph(r, a.Subgraph)) return Fail(r, "Choose a scene subgraph to instance");
                AddInstance(r, a.Subgraph, {.Name = std::string{NameStem(GetName(r, a.Subgraph))}});
            },
            [&](ToggleHidden) {
                for (const auto e : SelectedIn(r)) {
                    const auto target = SourceOfCopy(r, e);
                    if (!r.all_of<Hidden>(e)) Hide(r, target);
                    else Show(r, target);
                }
            },
            [&](const SetVisible &a) {
                ForEachObjectTarget(r, a.Target, [&](state::Entity e) {
                    if (!r.all_of<ObjectKind>(e) || r.all_of<SubElementOf>(e)) return;
                    if (a.Visible) Show(r, e);
                    else Hide(r, e); });
            },
            [&](const ShadeSelected &a) { append(r.get<const SelectionFlags>(state::Document).Meshes, graph::Shade{a.Mode, a.Angle}); },
            [&](const SetSelectedSharp &a) { append(edit_selection_meshes(a.Element), graph::MarkSharp{a.Sharp}); },
            [&](ParentToActive) {
                const auto active = FindActiveEntity(r);
                if (active == state::Null) return;
                std::vector<std::pair<state::Entity, state::Entity>> targets;
                for (const auto e : SelectedIn(r))
                    if (e != active) targets.push_back(CopyReferenceTarget(r, e, active));
                SortUnique(targets);
                for (const auto [entity, parent] : targets)
                    if (ParentCycle(r, entity, parent)) return Fail(r, "Loop in parents");
                const auto owned = r.view<const BodyPoseCache>() | std::ranges::to<std::vector>();
                for (const auto [entity, parent] : targets) SetParentKeepWorld(r, std::array{entity}, parent, owned);
            },
            [&](const ClearParent &a) {
                std::vector<state::Entity> targets;
                ForEachObjectTarget(r, a.Target, [&](state::Entity e) {
                    if (r.valid(e)) targets.push_back(e);
                });
                SortUnique(targets);
                ::ClearParents(r, targets);
            },
            [&](const SetParent &a) {
                if (!r.all_of<ObjectKind>(a.Entity) || !r.all_of<ObjectKind>(a.Parent) || r.any_of<SubElementOf, BoneIndex>(a.Entity) || r.any_of<SubElementOf, BoneIndex>(a.Parent)) return action::Fail(r, "Parenting requires two scene objects");
                const auto [entity, parent] = CopyReferenceTarget(r, a.Entity, a.Parent);
                if (ParentCycle(r, entity, parent)) return Fail(r, "Loop in parents");
                SetParentKeepWorld(r, std::array{entity}, parent, r.view<const BodyPoseCache>() | std::ranges::to<std::vector>());
            },
            [&](const AddEmpty &a) { ::AddEmpty(r, *a.Info); if (a.Info->BeginTranslate) begin_translate(); },
            [&](const AddArmature &a) {
                const auto entity = CreateArmatureObject(r, meshes, state::Null, a.Info->Name, a.Info->Transform, a.Info->Select, a.Info->Scene);
                graph::PlaceInGraph(r, a.Info->Graph, entity);
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
                const auto *material = r.valid(a.Material) ? r.try_get<const Material>(a.Material) : nullptr;
                if (!material) return Fail(r, "The material does not exist");
                r.Resources.get<GpuBuffers>().Materials.Update(std::as_bytes(std::span{&a.Value, 1}), uint64_t(material->Index) * sizeof(PBRMaterial) + a.Offset);
                r.Resources.get<RenderChanges>().Materials.emplace(a.Material);
                if constexpr (std::same_as<T, uint32_t>) {
                    // A texture slot is an unsigned field, and the document page shows it as a reference.
                    r.Resources.get<graph::GraphChanges>().Pages.emplace(state::Document);
                    // The export writes each texture slot's source texture, which the material's import record holds.
                    for (uint8_t slot = 0; slot < MTS_Count; ++slot) {
                        if (animation::TextureSlotOffset(slot) + offsetof(TextureInfo, Slot) != a.Offset) continue;
                        const auto texture = TextureAt(r, a.Value);
                        r.get_or_emplace<gltf::MaterialSourceMeta>(a.Material);
                        r.patch<gltf::MaterialSourceMeta>(a.Material, [&](auto &meta) { meta.TextureSlots[slot] = texture == state::Null ? InvalidSlot : r.get<const MaterializedTexture>(texture).SourceTextureIndex; });
                    }
                }
            },
            [&](const AddMaterial &a) {
                // A new material starts as a copy of the default record.
                const auto entity = ::AddMaterial(r, PBRMaterial{GetMaterials(r)[0]}, std::format("Material {}", r.view<const Material>().size() + 1u));
                r.Resources.get<RenderChanges>().Materials.emplace(entity);
                if (!a.PrimitiveIndex) return ::Select(r, entity);
                for_each_mesh_target(a.Target, [&](state::Entity e) { r.emplace_or_replace<MeshMaterialAssignment>(e, *a.PrimitiveIndex, entity); });
            },
            [&](const AddTexture &a) {
                if (const auto bytes = File::Read(project::ResolveAsset(r, a.Path)); !bytes) return Fail(r, std::format("Cannot read image '{}': {}", a.Path.string(), bytes.error()));
                const auto extension = a.Path.extension().string();
                const auto &sources = r.get_or_emplace<gltf::SourceAssets>(state::Document);
                const auto image = uint32_t(sources.Images.size()), texture = uint32_t(sources.Textures.size());
                const TextureParams params{
                    .ColorSpace = a.Color ? TextureColorSpace::Srgb : TextureColorSpace::Linear,
                    .WrapS = MTL::SamplerAddressModeRepeat,
                    .WrapT = MTL::SamplerAddressModeRepeat,
                    // A source texture without a sampler filters linearly across mip levels.
                    .Sampler = {.MinFilter = MTL::SamplerMinMagFilterLinear, .MagFilter = MTL::SamplerMinMagFilterLinear, .MipmapMode = MTL::SamplerMipFilterLinear, .UsesMipmaps = true},
                    .Name = a.Path.filename().string(),
                };
                r.patch<gltf::SourceAssets>(state::Document, [&](gltf::SourceAssets &sources) {
                    sources.Images.emplace_back(gltf::Image{.MimeType = extension == ".png" ? gltf::MimeType::PNG : extension == ".webp" ? gltf::MimeType::WEBP :
                                                                extension == ".ktx2"                                                     ? gltf::MimeType::KTX2 :
                                                                                                                                           gltf::MimeType::JPEG,
                                                            .Source = gltf::Image::SourceKind::External,
                                                            .Name = params.Name,
                                                            .SourcePath = project::AssetReference(r, a.Path).string()});
                    sources.Textures.emplace_back(gltf::Texture{.ImageIndex = image, .Name = params.Name});
                });
                ::Select(r, ::AddTexture(r, image, texture, params));
            },
            [&](const SetMaterialSlotSelection &a) {
                for_each_mesh_target(a.Target, [&](state::Entity e) { r.emplace_or_replace<MeshMaterialSlotSelection>(e, a.PrimitiveIndex); });
            },
            [&](const SetMaterialAssignment &a) {
                for_each_mesh_target(a.Target, [&](state::Entity e) { r.emplace_or_replace<MeshMaterialAssignment>(e, a.PrimitiveIndex, a.Material); });
            },
            [&](const SetProjection &a) {
                ForEachObjectTarget(r, a.Target, [&](state::Entity e) {
                    if (!HasLens(r, e)) return;
                    const float distance = std::max(Length(WorldTransformOf(r, e)->P), 1.f);
                    if (const auto *perspective = r.try_get<const Perspective>(e); perspective && a.Orthographic) SetLens(r, e, OrthographicFromPerspective(*perspective, distance));
                    else if (const auto *orthographic = r.try_get<const Orthographic>(e); orthographic && !a.Orthographic) SetLens(r, e, PerspectiveFromOrthographic(*orthographic, distance));
                });
            },
            [&](const SetLightType &a) {
                ForEachComponentTarget<SceneLight>(r, a.Target, [&](auto e) {
                    r.patch<SceneLight>(e, [&](auto &light) {
                        auto next = Defaults::MakeLight(a.Type);
                        next.Color = light.Color;
                        next.Intensity = light.Intensity;
                        next.Size = light.Size;
                        light = next;
                    });
                });
            },
            [&](const SetSpotCone &a) {
                ForEachComponentTarget<SceneLight>(r, a.Target, [&](auto e) {
                    r.patch<SceneLight>(e, [&](auto &light) {
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
