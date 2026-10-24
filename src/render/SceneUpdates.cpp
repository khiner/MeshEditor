#include "render/SceneUpdates.h"
#include "Parallel.h"
#include "Profile.h"
#include "SortUnique.h"
#include "armature/ArmatureComponents.h"
#include "gpu/LightRecord.h"
#include "mesh/MeshChanges.h"
#include "mesh/MeshComponents.h"
#include "metal/AutoreleaseScope.h"
#include "object/PendingSync.h"
#include "render/ClusterLodRepair.h"
#include "render/GpuBufferOps.h"
#include "render/GpuBuffers.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/LodNodeEdit.h"
#include "render/MaterialLodAttributes.h"
#include "render/MeshletBuild.h"
#include "render/MeshletBuildGpu.h"
#include "render/Pipelines.h"
#include "render/RenderChanges.h"
#include "render/RenderTargets.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionState.h"
#include "state/Registry.h"
#include "viewport/InteractionComponents.h"
#include "viewport/RenderExtent.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportConsumerFence.h"
#include "viewport/ViewportRenderGpu.h"

bool HiddenIn(const state::Registry &r, state::Entity e, state::Entity scene) { return r.all_of<Hidden>(e) || !InScene(r, e, scene); }

uint8_t InstanceStateBits(const state::Registry &r, state::Entity e) {
    return (r.all_of<Selected>(e) ? ElementStateSelected : 0) | (r.all_of<Active>(e) ? ElementStateActive : 0) | (r.all_of<Hidden>(e) ? InstanceStateHidden : 0);
}

namespace {
// The base state as the scene's states hold it, hidden when the instance is outside the scene.
uint8_t SceneState(const state::Registry &r, state::Entity e, state::Entity scene, uint8_t base) {
    return uint8_t(base | (InScene(r, e, scene) ? 0u : InstanceStateHidden));
}
} // namespace

void WriteInstanceState(const state::Registry &r, GpuBuffers &buffers, state::Entity e, uint32_t slot, uint8_t base) {
    for (uint32_t i = 0u; i < buffers.Scenes.size(); ++i) {
        buffers.Instances.States[i].Update(BytesOf(SceneState(r, e, buffers.Scenes[i], base)), uint64_t(slot) * sizeof(uint8_t));
    }
}

bool IsSilhouetteEligible(const state::Registry &r, state::Entity mesh_entity) {
    if (!r.valid(mesh_entity) || r.any_of<ObjectExtrasTag, ArmatureObject, BoneJoint>(mesh_entity) || !HasMesh(r, mesh_entity)) return false;
    return r.Resources.get<const MeshStore>().Arenas().FaceTriangles.Count(RecordOf(r, mesh_entity).FaceData) > 0u;
}

void RetallyMesh(state::Registry &r, state::Entity mesh_entity) {
    const auto id = DrawnStoreId(r, mesh_entity);
    auto &buffers = r.Resources.get<GpuBuffers>();
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &render = meshes.Render();
    const auto *mb = id ? meshes.TryGet(*id) : nullptr;
    if (!mb) return;
    const auto *models = r.try_get<const ModelsBuffer>(mesh_entity);
    const Range range = models ? Range{models->InstanceRange.Offset, models->InstanceCount} : Range{};
    const auto scene_states = buffers.Instances.States | std::views::transform([&](const mtl::Buffer &states) { return states.GetSpan<uint8_t>(range); }) | std::ranges::to<std::vector>();
    // An instance counts when some shown scene draws it.
    const auto drawn = [&](uint32_t i) { return std::ranges::any_of(scene_states, [i](std::span<const uint8_t> states) { return (states[i] & InstanceStateHidden) == 0u; }); };
    const auto flags = mb->RenderTopologies != 0u ? render.MeshRecords.Get({*id, 1u})[0].Display.Flags : 0u;
    buffers.Retally(*id, {
                             .Flags = flags,
                             .Instances = uint32_t(std::ranges::count_if(std::views::iota(0u, range.Count), drawn)),
                             .Nodes = meshes.LodNodeCount(*mb),
                             .Meshlets = meshes.MeshletCount(*mb),
                             .Depth = mb->LodDepth,
                             .Topologies = mb->RenderTopologies,
                             .SharpEdges = (flags & uint32_t(MeshletInstanceFlag::EditOverlay)) != 0u && meshes.GetEdgeSharpnessSummary(*id).Any,
                         });
}

void RepointMeshInstances(state::Registry &r, std::span<const state::Entity> mesh_entities) {
    const auto &buffers = r.Resources.get<GpuBuffers>();
    for (const auto mesh_entity : mesh_entities) {
        r.emplace_or_replace<MeshLayoutDirty>(mesh_entity);
        r.emplace_or_replace<MeshDisplayDirty>(mesh_entity);
        const auto *models = r.try_get<const ModelsBuffer>(mesh_entity);
        if (!models || !models->InstanceCount) continue;
        const auto first = models->InstanceRange.Offset;
        const auto ids = buffers.Instances.ObjectIdBuffer.GetSpan<uint32_t>({first, models->InstanceCount});
        const auto records = buffers.Instances.RecordBuffer.GetMutableSpan<InstanceRecord>({first, models->InstanceCount});
        const auto &mesh_buffers = RecordOf(r, mesh_entity);
        for (uint32_t i = 0u; i < ids.size(); ++i) {
            if (!ids[i]) continue;
            const auto instance_entity = r.EntityAt(ObjectIndex(ids[i]));
            if (instance_entity == state::Null) continue;
            const auto *ri = r.try_get<const RenderInstance>(instance_entity);
            if (!ri || ri->Entity != mesh_entity || ri->BufferIndex != first + i) continue;
            records[i].Mesh = mesh_buffers.RenderTopologies != 0u ? mesh_buffers.StoreId : InvalidOffset;
        }
        RetallyMesh(r, mesh_entity);
    }
}

bool RefreshMaterialLodAttributes(state::Registry &r) {
    const auto materials = GetMaterials(r);
    const auto debug = r.Resources.get<const GpuSceneState>().Displays.DebugChannels;
    std::vector<std::array<uint32_t, 2>> required(materials.size());
    for (size_t i = 0; i < materials.size(); ++i) required[i] = {MaterialLodAttributes(materials[i], debug, false), MaterialLodAttributes(materials[i], debug, true)};
    auto &cached = r.Resources.get<GpuSceneState>().RequiredMaterialAttributes;
    if (required == cached) return false;
    cached = std::move(required);
    return true;
}

void RefreshClusterLodAttributes(state::Registry &r, mtl::ComputeChain &chain, std::span<const state::Entity> mesh_entities) {
    auto &meshes = r.Resources.get<MeshStore>();
    auto &render = meshes.Render();
    const auto materials = GetMaterials(r);
    const auto debug = r.Resources.get<const GpuSceneState>().Displays.DebugChannels;
    std::vector<ClusterGroupSeeds> seeds;
    std::vector<uint32_t> changed;
    for (const auto entity : mesh_entities) {
        const auto *owner = TryRecordOf(r, entity);
        if (!owner) continue;
        if (!ClusterLodApplies((owner->RenderTopologies & 1u) != 0u, owner->Level0Count)) continue;
        const auto assignments = meshes.Arenas().PrimitiveMaterials.Get(owner->PrimitiveMaterials);
        const bool authored_tangents = (owner->CornerAttributes & MeshAttributeBit_Tangent) != 0u;
        const bool coarse = meshes.ClusterGroupCount(*owner) != 0u;
        changed.clear();
        meshes.ForEachPrimitive(*owner, [&](ElementHandle id, const PrimitiveRecord &primitive) {
            if (primitive.Topology != 0u) return;
            const uint32_t material = primitive.PrimitiveIndex < assignments.size() ? assignments[primitive.PrimitiveIndex] : 0u;
            const uint32_t required = MaterialLodAttributes(materials[material], debug, authored_tangents);
            if (required == primitive.LodAttributes) return;
            if (coarse) changed.push_back(id.Value);
            render.Primitives.GetMutable({render.Primitives.Address(owner->PrimitiveData, id).Value(), 1u})[0].LodAttributes = required;
        });
        if (changed.empty()) continue;
        std::ranges::sort(changed);
        auto &groups = seeds.emplace_back(ClusterGroupSeeds{.Entity = entity}).Groups;
        render.ClusterGroups.ForEach(owner->GroupData, [&](ElementHandle group, ElementAddress address) {
            const auto &links = render.GroupLinks.Get({address.Value(), 1u})[0];
            if (!links.MemberCount) return;
            const auto member = render.GroupClusterIds.Get(owner->GroupClusters.Values)[links.MemberOffset];
            const auto primitive = render.Meshlets.Get({render.Meshlets.Address(owner->MeshletData, ElementHandle{member}).Value(), 1u})[0].Primitive;
            if (std::ranges::binary_search(changed, primitive)) groups.push_back(group.Value);
        });
    }
    auto touched = InvalidateClusterGroups(r, seeds);
    std::vector<LodNodeRefit> refits;
    for (uint32_t i = 0u; i < seeds.size(); ++i) {
        touched[i].erase(std::ranges::unique(touched[i]).begin(), touched[i].end());
        refits.push_back(EditLodNodes(r, EditRecordOf(r, seeds[i].Entity), {}, {}, touched[i]));
    }
    RecordLodNodeRefits(r, chain, refits);
}

void BuildMeshlets(state::Registry &r, mtl::ComputeChain &chain, std::span<const state::Entity> mesh_entities, std::span<const state::Entity> bone_entities) {
    if (mesh_entities.empty() && bone_entities.empty()) return;
    for (const auto e : mesh_entities) ReleaseMeshEditWork(r, e);
    const profile::CpuScope scope{"BuildMeshlets"};
    auto &meshes = r.Resources.get<MeshStore>();
    std::vector<MeshletBuildSource> sources;
    sources.reserve(mesh_entities.size() * 3u + bone_entities.size());
    for (const auto entity : mesh_entities) {
        const auto mesh = GetDrawnMesh(r, entity);
        auto &mb = meshes.WriteRecord(mesh.GetStoreId());
        if (mesh.FaceCount()) sources.push_back({.Destination = &mb, .Topology = 0u, .ElementCount = mesh.TriangleIndexCount() / 3u});
        if (mesh.EdgeCount()) sources.push_back({.Destination = &mb, .Topology = 1u, .ElementCount = mesh.EdgeCount()});
        sources.push_back({.Destination = &mb, .Topology = 2u, .ElementCount = mesh.VertexCount()});
    }
    // Procedural bone geometry shares bounds, culling, routing, and indirect dispatch with standard meshlets.
    std::vector<state::Entity> bones;
    for (const auto entity : bone_entities) {
        auto &mb = EditRecordOf(r, entity);
        if (mb.ExtrasFaces.Count == 0u) continue;
        bones.push_back(entity);
        sources.push_back({
            .Destination = &mb,
            .Topology = 0u,
            .ElementCount = mb.ExtrasFaces.Count / 3u,
        });
    }
    BuildGpuMeshlets(r, chain, sources);
    RefreshClusterLodAttributes(r, chain, mesh_entities);
    RepointMeshInstances(r, mesh_entities);
    RepointMeshInstances(r, bones);
    for (const auto e : mesh_entities) r.emplace_or_replace<ClusterLodDemand>(e);
}

bool EditPinsFinest(const selection::PrimaryEditInstanceMap &primaries, const state::Registry &r, state::Entity mesh_entity) {
    return primaries.contains(mesh_entity) || r.all_of<MeshEditWork>(mesh_entity);
}

bool BuildDemandedClusterLods(state::Registry &r, mtl::ComputeChain &chain) {
    if (r.view<const ClusterLodDemand>().empty()) return false;
    auto &meshes = r.Resources.get<MeshStore>();
    auto &render = meshes.Render();
    // A mesh leaves the demand set once it has a hierarchy or its meshlets take none.
    // Arena offsets follow commit order, which is the demand set's entity order.
    std::vector<state::Entity> demanded;
    for (const auto entity : r.view<const ClusterLodDemand>() | std::ranges::to<std::vector>()) {
        const auto *mb = HasMesh(r, entity) ? TryRecordOf(r, entity) : nullptr;
        if (!mb || meshes.ClusterGroupCount(*mb) != 0u || !ClusterLodApplies(Mesh{meshes, mb->StoreId}.FaceCount() > 0u, meshes.MeshletCount(*mb))) r.remove<ClusterLodDemand>(entity);
        else demanded.push_back(entity);
    }
    if (r.get<const Interaction>(state::Document).Mode == InteractionMode::Edit && !demanded.empty()) {
        const auto &primaries = r.get<const EditPrimaries>(state::Document).All;
        std::erase_if(demanded, [&](state::Entity e) { return EditPinsFinest(primaries, r, e); });
    }
    if (demanded.empty()) return false;
    for (const auto entity : demanded) r.remove<ClusterLodDemand>(entity);
    const profile::CpuScope scope{"BuildClusterLods"};
    RefreshClusterLodAttributes(r, chain, demanded);
    const auto count = uint32_t(demanded.size());
    std::vector<MeshletBuildInputs> inputs;
    inputs.reserve(count);
    std::vector<std::vector<uint32_t>> live_triangles(count), live_primitive_counts(count);
    for (uint32_t i = 0u; i < count; ++i) {
        const auto entity = demanded[i];
        const auto &mb = RecordOf(r, entity);
        const auto mesh = GetDrawnMesh(r, entity);
        // After an edit, the build covers exactly the triangles the live finest clusters contain, per primitive in walk order.
        if (mb.MeshletRevision > 1u || mb.RenderTopologies != 1u) {
            auto &triangles = live_triangles[i];
            auto &primitive_counts = live_primitive_counts[i];
            meshes.ForEachPrimitive(mb, [&](ElementHandle, const PrimitiveRecord &primitive) {
                if (primitive.Topology != 0u) return;
                const auto start = triangles.size();
                if (primitive.LodFinestNode != InvalidOffset) {
                    meshes.ForEachLodMember(mb, render.LodNodes.Get({render.LodNodes.Address(mb.NodeData, ElementHandle{primitive.LodFinestNode}).Value(), 1u})[0], [&](uint32_t cluster) {
                        const auto &record = render.Meshlets.Get({render.Meshlets.Address(mb.MeshletData, ElementHandle{cluster}).Value(), 1u})[0];
                        const auto ids = render.MeshletTriangleIds.Get(mb.MeshletTriangles.Values).subspan(record.TriangleOffset, record.TriangleCount);
                        triangles.insert(triangles.end(), ids.begin(), ids.end());
                    });
                }
                primitive_counts.push_back(uint32_t(triangles.size() - start));
            });
        }
        auto corners = meshes.GetTriangleCorners(mesh.GetStoreId());
        corners.Handles = mb.MeshletRevision > 1u || mb.RenderTopologies != 1u ? std::span<const uint32_t>{live_triangles[i]} : render.MeshletTriangleIds.Get(mb.MeshletTriangles.Values);
        inputs.push_back(CaptureMeshletInputs(mesh, meshes, corners));
    }
    std::vector<ClusterLodBuild> lods(count);
    ParallelFor(count, [&](uint32_t i) { lods[i] = BuildMeshletClusterLod(meshes, RecordOf(r, demanded[i]), inputs[i], live_primitive_counts[i]); });
    // Finest IDs remain stable, so their canonical owner entries do too.
    std::vector<MeshStore::Record *> owners;
    owners.reserve(count);
    for (const auto entity : demanded) owners.push_back(&EditRecordOf(r, entity));
    CommitClusterLods(r, owners, lods);
    RepointMeshInstances(r, demanded);
    return true;
}

bool SyncShownScenes(state::Registry &r) {
    auto &buffers = r.Resources.get<GpuBuffers>();
    auto shown = ShownScenes(r);
    if (shown == buffers.Scenes) return false;
    // A scene that stays shown keeps its buffers, and the buffers of scenes no longer shown release with their slots.
    std::vector<mtl::Buffer> states, lights;
    states.reserve(shown.size());
    lights.reserve(shown.size());
    const auto slot_count = buffers.Instances.TransformBuffer.Count<Transform>();
    for (const auto scene : shown) {
        if (const auto it = std::ranges::find(buffers.Scenes, scene); it != buffers.Scenes.end()) {
            const auto i = size_t(it - buffers.Scenes.begin());
            states.emplace_back(std::move(buffers.Instances.States[i]));
            lights.emplace_back(std::move(buffers.Lights[i]));
            continue;
        }
        const auto bytes = states.emplace_back(buffers.Ctx, 0, SlotType::InstanceStateBuffer).SetCount<uint8_t>(slot_count);
        std::ranges::fill(bytes, uint8_t{0});
        for (const auto [e, render] : r.view<const RenderInstance>().each()) {
            if (render.BufferIndex != UINT32_MAX) bytes[render.BufferIndex] = SceneState(r, e, scene, InstanceStateBits(r, e));
        }
        lights.emplace_back(buffers.Ctx, sizeof(LightRecord), SlotType::LightBuffer);
    }
    buffers.Scenes = std::move(shown);
    buffers.Instances.States = std::move(states);
    buffers.Lights = std::move(lights);
    for (const auto mesh_entity : r.view<const ModelsBuffer>()) RetallyMesh(r, mesh_entity);
    return true;
}

uint32_t ShownSceneIndex(const state::Registry &r, state::Entity scene) {
    const auto &scenes = r.Resources.get<const GpuBuffers>().Scenes;
    const auto it = std::ranges::find(scenes, scene);
    return it != scenes.end() ? uint32_t(it - scenes.begin()) : 0u;
}

bool DeriveRenderInstances(state::Registry &r, bool rescope) {
    const auto &buffers = r.Resources.get<GpuBuffers>();
    const auto scene_states = buffers.Instances.States | std::views::transform([](const mtl::Buffer &states) { return states.GetMutableSpan<uint8_t>(); }) | std::ranges::to<std::vector>();
    std::vector<state::Entity> retallied;
    // A placed instance takes its hidden bit in each shown scene's states in place, keeping its slot and record.
    const auto write_hidden = [&](state::Entity e, const RenderInstance &render) {
        if (render.BufferIndex == UINT32_MAX) return;
        bool changed = false;
        for (uint32_t i = 0u; i < scene_states.size(); ++i) {
            auto &state = scene_states[i][render.BufferIndex];
            const auto next = uint8_t((state & ~InstanceStateHidden) | (HiddenIn(r, e, buffers.Scenes[i]) ? InstanceStateHidden : 0u));
            changed |= next != state;
            state = next;
        }
        if (changed) retallied.push_back(render.Entity);
    };
    for (const auto e : r.Resources.get<const RenderChanges>().InstanceVisibility.Entities) {
        if (!r.valid(e)) continue;
        const auto *instance = r.try_get<const Instance>(e);
        if (const auto *render = r.try_get<const RenderInstance>(e)) {
            if (instance && render->Entity == instance->Entity) {
                write_hidden(e, *render);
                continue;
            }
            r.remove<RenderInstance>(e);
        }
        if (instance) r.emplace<RenderInstance>(e, instance->Entity, UINT32_MAX);
    }
    if (rescope) {
        for (const auto [e, render] : r.view<const RenderInstance>().each()) write_hidden(e, render);
    }
    SortUnique(retallied);
    for (const auto mesh_entity : retallied) RetallyMesh(r, mesh_entity);
    return !retallied.empty();
}

namespace {
// Drawing-order index for a placed slot; order entries by descending object ID.
uint32_t SlotPosition(std::span<const uint32_t> list, std::span<const uint32_t> object_ids, uint32_t slot) {
    return uint32_t(std::ranges::lower_bound(list, object_ids[slot], std::ranges::greater{}, [&](uint32_t s) { return object_ids[s]; }) - list.begin());
}

void RebaseRenderInstances(state::Registry &r, std::span<const uint32_t> object_ids, Range slots) {
    for (uint32_t slot = slots.Offset; slot < slots.Offset + slots.Count; ++slot) r.edit<RenderInstance>(r.EntityAt(ObjectIndex(object_ids[slot]))).BufferIndex = slot;
}

// Merges the inserted slots into the drawing-order list by descending object ID.
void MergeInsertedSlots(GpuBuffers &buffers, std::vector<uint32_t> inserted) {
    auto &list = buffers.GpuInstanceSlots;
    const auto object_ids = buffers.Instances.ObjectIdBuffer.GetSpan<uint32_t>();
    std::ranges::sort(inserted, std::ranges::greater{}, [&](uint32_t slot) { return object_ids[slot]; });
    const auto old_count = list.Count<uint32_t>();
    const auto slots = list.SetCount<uint32_t>(old_count + uint32_t(inserted.size()));
    std::ranges::copy(inserted, slots.begin() + old_count);
    std::ranges::inplace_merge(slots, slots.begin() + old_count, std::ranges::greater{}, [&](uint32_t s) { return object_ids[s]; });
}
} // namespace

SyncResult SyncModelsBuffers(state::Registry &r) {
    const profile::CpuScope scope{"SyncModelsBuffers"};
    auto &buffers = r.Resources.get<GpuBuffers>();
    auto &meshes = r.Resources.get<MeshStore>();
    auto &scene = r.Resources.get<GpuSceneState>();
    // Remove released records from flag totals before building new records.
    for (const auto id : meshes.TakeReleased()) buffers.Retally(id, {});
    std::vector<state::Entity> new_mesh_entities, new_extras_entities;
    std::vector<uint8_t> listed;
    for (auto e : r.Resources.get<const MeshChanges>().NewBufferEntity) {
        if (!r.valid(e)) continue;
        // A record keeps its render data across handle changes that leave it intact.
        const auto id = DrawnStoreId(r, e);
        if (!id || meshes.Get(*id).RenderTopologies != 0u) continue;
        if (*id >= listed.size()) listed.resize(*id + 1u);
        if (std::exchange(listed[*id], uint8_t{1})) continue;
        if (HasMesh(r, e)) new_mesh_entities.emplace_back(e);
        else if (r.all_of<ObjectExtrasTag>(e) || r.all_of<ArmatureObject>(e) || r.all_of<BoneJoint>(e)) new_extras_entities.emplace_back(e);
    }

    const auto relayout = [&](state::Entity buffer_entity) { r.emplace_or_replace<MeshLayoutDirty>(buffer_entity); };

    // Remove destroyed instance slots and compact each surviving owner buffer once.
    bool slots_changed = false;
    if (auto *pending = r.Resources.find<PendingSlotRemovals>()) {
        auto &removals = pending->Instances;
        auto &retired = pending->Retired;
        std::ranges::sort(retired);
        SortUnique(removals);
        const auto object_ids = buffers.Instances.ObjectIdBuffer.GetSpan<uint32_t>();
        const auto list = buffers.GpuInstanceSlots.GetMutableSpan<uint32_t>();
        // Each removed slot's drawing-order entry, found by the ID it contains before its buffer compacts.
        std::vector<uint32_t> erased_positions;
        erased_positions.reserve(removals.size());
        for (const auto &removal : removals) erased_positions.push_back(SlotPosition(list, object_ids, removal.Index));
        std::erase_if(removals, [&](const auto &removal) {
            return !r.valid(removal.Owner) || std::ranges::binary_search(retired, removal.Owner) || !r.all_of<ModelsBuffer>(removal.Owner);
        });
        auto by_owner = removals | std::views::chunk_by([](const auto &a, const auto &b) { return a.Owner == b.Owner; }); // NOLINT(misc-const-correctness)
        // A survivor past an owner's first erased slot moves down by the slots erased before it.
        // Every entry is found before any buffer compacts, while each slot still contains its ID.
        struct SlotMove {
            uint32_t Position, Slot;
        };
        std::vector<SlotMove> moves;
        for (const auto erased : by_owner) {
            const auto &mb = r.get<const ModelsBuffer>(erased.front().Owner);
            uint32_t shift = 0u;
            auto next = erased.begin();
            for (auto slot = erased.front().Index; slot < mb.InstanceRange.Offset + mb.InstanceCount; ++slot) {
                if (next != erased.end() && next->Index == slot) {
                    ++shift;
                    ++next;
                    continue;
                }
                moves.push_back({SlotPosition(list, object_ids, slot), slot - shift});
            }
        }
        std::vector<std::pair<state::Entity, Range>> compacted;
        for (const auto erased : by_owner) {
            const auto owner = erased.front().Owner;
            auto &mb = r.edit<ModelsBuffer>(owner);
            buffers.Instances.CompactErase({mb.InstanceRange.Offset, mb.InstanceCount}, erased, &PendingSlotRemovals::Removal::Index);
            mb.InstanceCount -= uint32_t(std::ranges::size(erased));
            const auto first = erased.front().Index;
            compacted.emplace_back(owner, Range{first, mb.InstanceRange.Offset + mb.InstanceCount - first});
        }
        for (const auto &move : moves) list[move.Position] = move.Slot;
        for (const auto position : erased_positions) list[position] = InvalidOffset;
        buffers.GpuInstanceSlots.SetCount<uint32_t>(uint32_t(std::ranges::remove(list, InvalidOffset).begin() - list.begin()));
        for (const auto &[owner, moved] : compacted) {
            RebaseRenderInstances(r, object_ids, moved);
            relayout(owner);
            RetallyMesh(r, owner);
        }
        for (const auto owner : retired) {
            if (!r.valid(owner)) continue;
            relayout(owner);
            RetallyMesh(r, owner);
        }
        slots_changed = !erased_positions.empty();
        removals.clear();
        retired.clear();
    }

    // New instances by buffer, in entity order within each buffer.
    std::vector<std::pair<state::Entity, state::Entity>> placements;
    for (auto entity : r.Resources.get<const RenderChanges>().InstanceCreated) {
        if (!r.valid(entity)) continue;
        if (const auto *ri = r.try_get<const RenderInstance>(entity); ri && ri->BufferIndex == UINT32_MAX) placements.emplace_back(ri->Entity, entity);
    }
    std::ranges::sort(placements);
    auto groups = placements | std::views::chunk_by([](const auto &a, const auto &b) { return a.first == b.first; }); // NOLINT(misc-const-correctness)
    // Each buffer without a range or without room takes a new range, every one carved from a single allocation in buffer order.
    const auto new_capacity = [&](state::Entity buffer_entity, uint32_t n) {
        const auto *models = r.try_get<const ModelsBuffer>(buffer_entity);
        if (!models) return n;
        const auto total = models->InstanceCount + n;
        return total > models->InstanceRange.Count ? std::max(models->InstanceRange.Count * 2, total) : 0u;
    };
    uint32_t capacity = 0u;
    for (const auto group : groups) capacity += new_capacity(group.front().first, uint32_t(std::ranges::size(group)));
    auto block = buffers.Instances.Allocate(capacity);
    const auto carve = [&](uint32_t count) {
        const Range range{block.Offset, count};
        block.Offset += count;
        return range;
    };
    // Return inserted instances so the settle pass writes their transform slots before submission.
    std::vector<state::Entity> newly_inserted;
    newly_inserted.reserve(placements.size());
    std::vector<uint32_t> object_ids, inserted_slots;
    std::vector<uint8_t> bases, states;
    std::vector<InstanceRecord> instance_records;
    for (const auto group : groups) {
        const auto buffer_entity = group.front().first;
        const auto n = uint32_t(std::ranges::size(group));
        const auto grown = new_capacity(buffer_entity, n);
        // Defer ModelsBuffer creation until its initial capacity is known.
        if (!r.all_of<ModelsBuffer>(buffer_entity)) r.emplace<ModelsBuffer>(buffer_entity, ModelsBuffer{carve(grown), 0});
        auto &mb = r.edit<ModelsBuffer>(buffer_entity);
        const auto new_total = mb.InstanceCount + n;
        if (new_total > mb.InstanceRange.Count) {
            const auto old_range = mb.InstanceRange;
            mb.InstanceRange = carve(grown);
            buffers.Instances.CopyInstances(old_range.Offset, mb.InstanceRange.Offset, mb.InstanceCount);
            // The moved slots keep their drawing-order entries, found by the IDs both ranges contain.
            const auto ids = buffers.Instances.ObjectIdBuffer.GetSpan<uint32_t>();
            const auto list = buffers.GpuInstanceSlots.GetMutableSpan<uint32_t>();
            for (uint32_t i = 0u; i < mb.InstanceCount; ++i) list[SlotPosition(list, ids, old_range.Offset + i)] = mb.InstanceRange.Offset + i;
            RebaseRenderInstances(r, ids, {mb.InstanceRange.Offset, mb.InstanceCount});
            buffers.Instances.Free(old_range);
            slots_changed = true;
        }
        object_ids.resize(n);
        bases.resize(n);
        states.resize(n);
        instance_records.assign(n, {});
        const auto base_index = mb.InstanceRange.Offset + mb.InstanceCount;
        const auto *mesh_buffers = TryRecordOf(r, buffer_entity);
        const auto mesh_record = mesh_buffers && mesh_buffers->RenderTopologies != 0u ? mesh_buffers->StoreId : InvalidOffset;
        for (uint32_t j = 0; j < n; ++j) {
            const auto instance_entity = group[j].second;
            r.edit<RenderInstance>(instance_entity).BufferIndex = base_index + j;
            inserted_slots.push_back(base_index + j);
            object_ids[j] = ObjectId(instance_entity);
            bases[j] = InstanceStateBits(r, instance_entity);
            instance_records[j] = {
                .Mesh = mesh_record,
                .ObjectId = ObjectId(instance_entity),
            };
        }
        // Transform slots stay unwritten here, and the world-transform recompute and the bone display pass write them before submit.
        buffers.Instances.ObjectIdBuffer.Update(BytesOf(object_ids), uint64_t(base_index) * sizeof(uint32_t));
        for (uint32_t i = 0u; i < buffers.Scenes.size(); ++i) {
            for (uint32_t j = 0; j < n; ++j) states[j] = SceneState(r, group[j].second, buffers.Scenes[i], bases[j]);
            buffers.Instances.States[i].Update(BytesOf(states), uint64_t(base_index) * sizeof(uint8_t));
        }
        buffers.Instances.RecordBuffer.Update(BytesOf(instance_records), uint64_t(base_index) * sizeof(InstanceRecord));
        // Bounds reduction populates mesh instances; extras retain empty bounds and bypass culling.
        std::ranges::fill(buffers.Instances.GetMutableBounds({base_index, n}), AABB{});
        mb.InstanceCount = new_total;
        for (const auto &[_, instance_entity] : group) newly_inserted.push_back(instance_entity);
        relayout(buffer_entity);
        RetallyMesh(r, buffer_entity);
    }
    if (!inserted_slots.empty()) MergeInsertedSlots(buffers, std::move(inserted_slots));
    slots_changed |= !newly_inserted.empty();
    if (slots_changed) scene.OverlayJobsDirty = true;
    return {std::move(newly_inserted), std::move(new_mesh_entities), std::move(new_extras_entities), slots_changed};
}

bool SyncViewportRenderResources(state::Registry &r, state::Entity viewport_entity, uvec2 extent) {
    const auto render_extent = std::bit_cast<mtl::Extent2D>(extent);
    if (render_extent.Width == 0 || render_extent.Height == 0) return false;
    const bool want_transmission = RealTransmission(r, viewport_entity, r.get<const ViewportDisplay>(viewport_entity), r.Resources.get<const GpuSceneState>().MeshPbrFeatures);
    auto &viewport = r.edit<Viewport>(viewport_entity);
    auto &targets = viewport.Targets;

    const auto &ctx = r.Resources.get<const mtl::Context>();
    const auto &samplers = viewport.Samplers;
    auto &slots = r.Resources.get<mtl::BindlessSet>();
    const auto set_sampler = [&](uint32_t slot, SampledTexture sampled) { slots.SetSampler({SlotType::Sampler, slot}, sampled.Texture, sampled.Sampler); };
    if (targets.BuiltColorExtent() == render_extent) {
        if (!targets.EnsureTransmissionResources(ctx, render_extent, want_transmission)) return false;
        set_sampler(samplers.Transmission, targets.TransmissionSampler());
        return true;
    }

    // Wait for the live consumer (ImGui) to finish sampling the old resources before recreating them.
    if (auto *consumer = r.Resources.get<const ViewportConsumerFence>().Value) {
        const mtl::AutoreleaseScope native_scope;
        consumer->waitUntilCompleted();
    }
    targets.SetExtent(ctx, render_extent, slots);
    targets.EnsureTransmissionResources(ctx, render_extent, want_transmission);
    {
        const profile::CpuScope scope{"UpdateSamplerSlots"};
        set_sampler(samplers.Silhouette, targets.Nearest(&targets.Screen->SilhouetteImage));
        set_sampler(samplers.SceneColor, targets.SceneColorSampler());
        set_sampler(samplers.OverlayColor, targets.OverlayColorSampler());
        set_sampler(samplers.Transmission, targets.TransmissionSampler());
        set_sampler(samplers.MotionBlurOutput, targets.MotionBlurOutputSampler());
        set_sampler(samplers.Velocity, targets.Nearest(nullptr));
        set_sampler(samplers.SceneDepth, targets.SceneDepthSampler());
        set_sampler(samplers.DepthPyramid, targets.DepthPyramidSampler());
        set_sampler(samplers.OutlineOccluderPyramid, targets.OutlineOccluderPyramidSampler());
    }
    return true;
}
