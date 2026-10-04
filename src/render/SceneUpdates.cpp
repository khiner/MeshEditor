#include "render/SceneUpdates.h"
#include "Parallel.h"
#include "Profile.h"
#include "SortUnique.h"
#include "armature/ArmatureComponents.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "metal/AutoreleaseScope.h"
#include "object/PendingSync.h"
#include "render/GpuBufferOps.h"
#include "render/GpuBuffers.h"
#include "gpu/BoundsEntry.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/MeshletBuild.h"
#include "render/MeshletBuildGpu.h"
#include "render/ClusterLodRepair.h"
#include "render/MaterialLodAttributes.h"
#include "render/LodNodeEdit.h"
#include "metal/Dispatch.h"
#include "render/Pipelines.h"
#include "render/RenderTargets.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionState.h"
#include "state/Scene.h"
#include "viewport/RenderExtent.h"
#include "viewport/ViewportConsumerFence.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportRenderGpu.h"

using state::Change;
uint8_t InstanceStateBits(const state::Scene &r, state::Entity e) {
    return (r.all_of<Selected>(e) ? ElementStateSelected : 0) | (r.all_of<Active>(e) ? ElementStateActive : 0) | (r.all_of<Hidden>(e) ? InstanceStateHidden : 0);
}

bool IsSilhouetteEligible(const state::Scene &r, state::Entity mesh_entity) {
    if (!r.valid(mesh_entity) || r.any_of<ObjectExtrasTag, ArmatureObject, BoneJoint>(mesh_entity)) return false;
    const auto mesh = TryGetMesh(r, mesh_entity);
    return mesh && mesh->FaceCount() > 0;
}

void RetallyMesh(state::Scene &r, state::Entity mesh_entity) {
    const auto id = DrawnStoreId(r, mesh_entity);
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto &meshes = r.Context.get<const MeshStore>();
    auto &render = meshes.Render();
    const auto *mb = id ? meshes.TryGet(*id) : nullptr;
    if (!mb) return;
    const auto *models = r.try_get<const ModelsBuffer>(mesh_entity);
    const auto states = models ? buffers.Instances.StateBuffer.GetSpan<uint8_t>({models->InstanceRange.Offset, models->InstanceCount}) : std::span<const uint8_t>{};
    buffers.Retally(*id, {
        .Flags = mb->RenderTopology != InvalidOffset ? render.MeshRecords.Get({*id, 1u})[0].Display.Flags : 0u,
        .Instances = uint32_t(std::ranges::count_if(states, [](uint8_t state) { return (state & InstanceStateHidden) == 0u; })),
        .Nodes = render.ActiveMeshlets.Count(mb->NodeRoot),
        .Meshlets = meshes.MeshletCount(*mb),
        .Depth = mb->LodDepth,
    });
}

// Assign placed primitives to the affected meshes' instance ranges.
void RepointMeshInstances(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    auto &buffers = r.Context.get<GpuBuffers>();
    auto &scene = r.Context.get<GpuSceneState>();
    for (const auto mesh_entity : mesh_entities) {
        scene.DisplayDirty.insert(mesh_entity);
        const auto *models = r.try_get<const ModelsBuffer>(mesh_entity);
        if (!models || !models->InstanceCount) continue;
        const auto first = models->InstanceRange.Offset;
        const auto ids = buffers.Instances.ObjectIdBuffer.GetSpan<uint32_t>({first,models->InstanceCount});
        auto records = buffers.Instances.RecordBuffer.GetMutableSpan<InstanceRecord>({first,models->InstanceCount});
        const auto &mesh_buffers = RecordOf(r, mesh_entity);
        for (uint32_t i=0u; i<ids.size(); ++i) {
            if (!ids[i]) continue;
            const auto instance_entity = r.EntityAt(ObjectIndex(ids[i]));
            if (instance_entity==state::Null) continue;
            const auto *ri = r.try_get<const RenderInstance>(instance_entity);
            if (!ri || ri->Entity!=mesh_entity || ri->BufferIndex!=first+i) continue;
            records[i].Mesh = mesh_buffers.RenderTopology != InvalidOffset ? mesh_buffers.StoreId : InvalidOffset;
        }
        RetallyMesh(r, mesh_entity);
    }
}

bool RepointChangedMeshes(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    auto &buffers = r.Context.get<GpuBuffers>();
    auto &scene = r.Context.get<GpuSceneState>();
    RepointMeshInstances(r, mesh_entities);
    bool changed = false;
    for (const auto entity : mesh_entities) {
        const auto *models = r.try_get<const ModelsBuffer>(entity);
        if (!models || !models->InstanceCount) continue;
        const auto &owner = RecordOf(r, entity);
        const auto work = scene.EditWork.find(entity);
        changed = changed || scene.PosedByEntity.contains(entity) || (work != scene.EditWork.end() && work->second.RequiresPose) ||
            buffers.DrawsNewTopology(owner.RenderTopology);
    }
    return changed;
}

namespace {
// The debug channel material-shaded viewports show, which the LOD attributes preserve.
DebugChannel LodDebugChannel(const state::Scene &r) {
    DebugChannel debug = DebugChannel::None;
    for (const auto [_, display] : r.view<const ViewportDisplay>().each())
        if (!WorkbenchShading(display.ViewportShading)) debug = display.DebugChannel;
    return debug;
}
} // namespace

bool RefreshMaterialLodAttributes(state::Scene &r) {
    const auto materials = GetMaterials(r);
    const auto debug = LodDebugChannel(r);
    std::vector<std::array<uint32_t, 2>> required(materials.size());
    for (size_t i = 0; i < materials.size(); ++i) required[i] = {MaterialLodAttributes(materials[i], debug, false), MaterialLodAttributes(materials[i], debug, true)};
    auto &cached = r.Context.get<GpuSceneState>().RequiredMaterialAttributes;
    if (required == cached) return false;
    cached = std::move(required);
    return true;
}

void RefreshClusterLodAttributes(state::Scene &r, mtl::ComputeChain &chain, std::span<const state::Entity> mesh_entities) {
    auto &meshes = r.Context.get<MeshStore>();
    auto &render = meshes.Render();
    const auto materials = GetMaterials(r);
    const auto debug = LodDebugChannel(r);
    // Each coarse owner's groups over primitives whose attributes changed.
    std::vector<ClusterGroupSeeds> seeds;
    std::vector<uint32_t> changed;
    for (const auto entity : mesh_entities) {
        const auto *owner = TryRecordOf(r, entity);
        if (!owner) continue;
        if (!ClusterLodApplies(owner->RenderTopology == uint32_t(MeshPrimitiveTopology::Triangle), owner->Level0Count)) continue;
        const auto assignments = meshes.Arenas().PrimitiveMaterials.Get(owner->PrimitiveMaterials);
        const bool authored_tangents = (owner->CornerAttributes & MeshAttributeBit_Tangent) != 0u;
        const bool coarse = meshes.ClusterGroupCount(*owner) != 0u;
        changed.clear();
        meshes.ForEachPrimitive(*owner, [&](uint32_t id, const PrimitiveRecord &primitive) {
            const uint32_t material = primitive.PrimitiveIndex < assignments.size() ? assignments[primitive.PrimitiveIndex] : 0u;
            const uint32_t required = MaterialLodAttributes(materials[material], debug, authored_tangents);
            if (required == primitive.LodAttributes) return;
            if (coarse) changed.push_back(id);
            render.Primitives.GetMutable({id, 1u})[0].LodAttributes = required;
        });
        if (changed.empty()) continue;
        std::ranges::sort(changed);
        auto &groups = seeds.emplace_back(ClusterGroupSeeds{.Entity = entity}).Groups;
        render.ActiveMeshlets.ForEach(owner->GroupRoot, [&](uint32_t group) {
            const auto &links = render.GroupLinks.Get({group, 1u})[0];
            if (!links.MemberCount) return;
            const auto member = render.GroupClusterIds.Get({links.MemberOffset, 1u})[0];
            const auto primitive = render.Meshlets.Get({member, 1u})[0].Primitive;
            if (std::ranges::binary_search(changed, primitive)) groups.push_back(group);
        });
    }
    auto touched = InvalidateClusterGroups(r, seeds);
    std::vector<LodNodeRefit> refits;
    for (uint32_t i = 0u; i < seeds.size(); ++i) {
        touched[i].erase(std::unique(touched[i].begin(), touched[i].end()), touched[i].end());
        refits.push_back(EditLodNodes(r, chain, EditRecordOf(r, seeds[i].Entity), {}, {}, touched[i]));
    }
    RecordLodNodeRefits(r, chain, refits);
}

void BuildMeshlets(state::Scene &r, mtl::ComputeChain &chain, std::span<const state::Entity> mesh_entities, std::span<const state::Entity> bone_entities) {
    if (mesh_entities.empty() && bone_entities.empty()) return;
    for (auto e : mesh_entities) ReleaseMeshEditWork(r, e);
    const profile::CpuScope scope{"BuildMeshlets"};
    auto &buffers = r.Context.get<GpuBuffers>();
    buffers.PreludeStale = true;
    auto &meshes = r.Context.get<MeshStore>();
    // A record changing topology releases its former render ranges.
    std::vector<MeshStore::Record *> retopologized;
    for (const auto entity : mesh_entities) {
        const auto mesh = GetMesh(r, entity);
        auto &mb = meshes.WriteRecord(mesh.GetStoreId());
        if (mb.RenderTopology == InvalidOffset || mb.RenderTopology == mesh.PrimitiveTopology()) continue;
        retopologized.push_back(&mb);
    }
    if (!retopologized.empty()) meshes.ReleaseRender(retopologized);
    std::vector<MeshletBuildSource> sources;
    sources.reserve(mesh_entities.size() + bone_entities.size());
    for (const auto entity : mesh_entities) {
        const auto mesh = GetMesh(r, entity);
        const auto topology = mesh.PrimitiveTopology();
        const bool faces = topology == 0u, lines = topology == 1u;
        auto &mb = meshes.WriteRecord(mesh.GetStoreId());
        sources.push_back({
            .Destination = &mb,
            .Topology = topology,
            .ElementCount = faces ? mesh.TriangleIndexCount() / 3u : lines ? mesh.EdgeCount() : mesh.VertexCount(),
        });
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
    r.Context.get<GpuSceneState>().LodDemand.insert(mesh_entities.begin(), mesh_entities.end());
}

bool EditPinsFinest(const selection::PrimaryEditInstanceMap &primaries, const GpuSceneState &scene, state::Entity mesh_entity) {
    return primaries.contains(mesh_entity) || scene.EditWork.contains(mesh_entity);
}

bool BuildDemandedClusterLods(state::Scene &r, state::Entity viewport) {
    auto &scene = r.Context.get<GpuSceneState>();
    if (scene.LodDemand.empty()) return false;
    auto &buffers = r.Context.get<GpuBuffers>();
    auto &meshes = r.Context.get<MeshStore>();
    auto &render = meshes.Render();
    // A mesh leaves the demand set once it has a hierarchy or its meshlets take none.
    std::erase_if(scene.LodDemand, [&](state::Entity entity) {
        const auto *handle = r.valid(entity) ? r.try_get<const MeshHandle>(entity) : nullptr;
        const auto *mb = handle ? meshes.TryGet(handle->StoreId) : nullptr;
        return !mb || meshes.ClusterGroupCount(*mb) != 0u || !ClusterLodApplies(Mesh{meshes, handle->StoreId}.FaceCount() > 0u, meshes.MeshletCount(*mb));
    });
    std::vector<state::Entity> demanded{scene.LodDemand.begin(), scene.LodDemand.end()};
    if (r.get<const Interaction>(viewport).Mode == InteractionMode::Edit && !demanded.empty()) {
        const auto &primaries = r.get<const EditPrimaries>(viewport).All;
        std::erase_if(demanded, [&](state::Entity e) { return EditPinsFinest(primaries, scene, e); });
    }
    if (demanded.empty()) return false;
    for (const auto entity : demanded) scene.LodDemand.erase(entity);
    // Arena offsets follow commit order.
    std::ranges::sort(demanded);
    const profile::CpuScope scope{"BuildClusterLods"};
    {
        mtl::ComputeChain chain{buffers.Ctx};
        RefreshClusterLodAttributes(r, chain, demanded);
        chain.Submit();
    }
    const uint32_t count = uint32_t(demanded.size());
    std::vector<MeshletBuildInputs> inputs;
    inputs.reserve(count);
    std::vector<std::vector<uint32_t>> live_triangles(count), live_primitive_counts(count);
    for (uint32_t i=0u;i<count;++i) {
        const auto entity=demanded[i];
        const auto &mb = RecordOf(r,entity);
        const auto mesh=GetMesh(r,entity);
        // After an edit, the build covers exactly the triangles the live finest clusters hold, per primitive in walk order.
        if (mb.MeshletRevision>1u) {
            auto &triangles=live_triangles[i];
            auto &primitive_counts=live_primitive_counts[i];
            meshes.ForEachPrimitive(mb,[&](uint32_t, const PrimitiveRecord &primitive) {
                const auto root=primitive.LodFinestNode==InvalidOffset ? InvalidOffset :
                    render.LodNodes.Get({primitive.LodFinestNode,1u})[0].MeshletRoot;
                const auto start=triangles.size();
                render.ActiveMeshlets.ForEach(root,[&](uint32_t cluster) {
                    const auto &record=render.Meshlets.Get({cluster,1u})[0];
                    const auto ids=render.MeshletTriangleIds.Get({record.TriangleOffset,record.TriangleCount});
                    triangles.insert(triangles.end(),ids.begin(),ids.end());
                });
                primitive_counts.push_back(uint32_t(triangles.size()-start));
            });
            inputs.push_back(CaptureMeshletInputs(mesh,meshes,
                {meshes.Arenas().Triangles.Buffer.GetSpan<uvec3>(),triangles}));
        } else inputs.push_back(CaptureMeshletInputs(mesh,meshes,
            {meshes.Arenas().Triangles.Buffer.GetSpan<uvec3>(),render.MeshletTriangleIds.Get(mb.MeshletTriangles)}));
    }
    std::vector<ClusterLodBuild> lods(count);
    ParallelFor(count, [&](uint32_t i) { lods[i] = BuildMeshletClusterLod(meshes, RecordOf(r, demanded[i]), inputs[i],live_primitive_counts[i]); });
    // Finest IDs remain stable, so their canonical owner entries do too.
    std::vector<MeshStore::Record *> owners;
    owners.reserve(count);
    for (const auto entity : demanded) owners.push_back(&EditRecordOf(r,entity));
    CommitClusterLods(r,owners,lods);
    RepointMeshInstances(r, demanded);
    buffers.PreludeStale = true;
    return true;
}

bool DeriveRenderInstances(state::Scene &r) {
    auto &buffers = r.Context.get<GpuBuffers>();
    std::span<uint8_t> states;
    std::vector<state::Entity> retallied;
    for (const auto e : reactive(r, state::Change::InstanceVisibility).Entities) {
        if (!r.valid(e)) continue;
        const auto *instance = r.try_get<const Instance>(e);
        if (const auto *render = r.try_get<const RenderInstance>(e)) {
            if (instance && render->Entity == instance->Entity) {
                // A placed instance takes its Hidden bit in place, keeping its slot and record.
                if (render->BufferIndex == UINT32_MAX) continue;
                if (states.empty()) states = buffers.Instances.GetMutableStates();
                const bool hidden = r.all_of<Hidden>(e);
                auto &state = states[render->BufferIndex];
                const auto next = uint8_t((state & ~InstanceStateHidden) | (hidden ? InstanceStateHidden : 0u));
                if (next == state) continue;
                state = next;
                retallied.push_back(render->Entity);
                continue;
            }
            r.remove<RenderInstance>(e);
        }
        if (instance) r.emplace<RenderInstance>(e, instance->Entity, UINT32_MAX);
    }
    SortUnique(retallied);
    for (const auto mesh_entity : retallied) RetallyMesh(r, mesh_entity);
    return !retallied.empty();
}

namespace {
// A placed slot's entry in the drawing-order list, which holds every placed slot once by descending object ID.
uint32_t SlotPosition(std::span<const uint32_t> list, std::span<const uint32_t> object_ids, uint32_t slot) {
    return uint32_t(std::ranges::lower_bound(list, object_ids[slot], std::ranges::greater{}, [&](uint32_t s) { return object_ids[s]; }) - list.begin());
}

// Points the render instance of each slot in the range at its slot.
void RebaseRenderInstances(state::Scene &r, std::span<const uint32_t> object_ids, Range slots) {
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

SyncResult SyncModelsBuffers(state::Scene &r) {
    const profile::CpuScope scope{"SyncModelsBuffers"};
    auto &buffers = r.Context.get<GpuBuffers>();
    auto &meshes = r.Context.get<MeshStore>();
    auto &scene = r.Context.get<GpuSceneState>();
    // Released records leave the flag totals ahead of this pass's builds.
    for (const auto id : meshes.TakeReleased()) buffers.Retally(id, {});
    std::vector<state::Entity> new_mesh_entities, new_extras_entities;
    std::vector<uint32_t> new_ids;
    for (auto e : reactive(r, Change::NewBufferEntity)) {
        if (!r.valid(e)) continue;
        // A record keeps its render data across handle changes that leave it intact.
        const auto id = DrawnStoreId(r, e);
        if (!id || meshes.Get(*id).RenderTopology != InvalidOffset || std::ranges::contains(new_ids, *id)) continue;
        new_ids.push_back(*id);
        if (HasMesh(r, e)) new_mesh_entities.emplace_back(e);
        else if (r.all_of<ObjectExtrasTag>(e) || r.all_of<ArmatureObject>(e) || r.all_of<BoneJoint>(e)) new_extras_entities.emplace_back(e);
    }

    bool layout = false;
    const auto entries = buffers.BoundsReduceEntries.GetMutableSpan<BoundsEntry>();
    // A mesh run without per-instance deformation follows its instance range in place, and any other mesh lays out again.
    const auto update_run = [&](state::Entity buffer_entity, bool inserted) {
        const auto run = scene.BoundsRuns.find(buffer_entity);
        const auto *models = r.try_get<const ModelsBuffer>(buffer_entity);
        if (run == scene.BoundsRuns.end()) {
            layout |= models && HasMesh(r, buffer_entity);
            return;
        }
        const auto [first, count, posed] = run->second;
        if (!models || posed) {
            // An entry without instances writes no instance bounds, so a retired run stays inert until the next layout.
            for (uint32_t i = first; i < first + count && i < entries.size(); ++i) entries[i].InstanceCount = 0u;
            layout |= posed && models;
            return;
        }
        auto &entry = entries[first];
        entry.FirstInstance = models->InstanceRange.Offset;
        entry.InstanceCount = models->InstanceCount;
        if (inserted) scene.DirtyBoundsEntries.push_back(first);
        if (const auto *mb = TryRecordOf(r, buffer_entity); inserted && mb && buffers.DrawsNewTopology(mb->RenderTopology)) layout = true;
    };

    // Destroyed render instances leave their slots, and each owner keeping its buffer compacts once.
    bool slots_changed = false;
    if (auto *pending = r.Context.find<PendingSlotRemovals>()) {
        auto &removals = pending->Instances;
        auto &retired = pending->Retired;
        std::ranges::sort(retired);
        SortUnique(removals);
        const auto object_ids = buffers.Instances.ObjectIdBuffer.GetSpan<uint32_t>();
        const auto list = buffers.GpuInstanceSlots.GetMutableSpan<uint32_t>();
        // Each removed slot's drawing-order entry, found by the ID it holds before its buffer compacts.
        std::vector<uint32_t> erased_positions;
        erased_positions.reserve(removals.size());
        for (const auto &removal : removals) erased_positions.push_back(SlotPosition(list, object_ids, removal.Index));
        std::erase_if(removals, [&](const auto &removal) {
            return !r.valid(removal.Owner) || std::ranges::binary_search(retired, removal.Owner) || !r.all_of<ModelsBuffer>(removal.Owner);
        });
        auto by_owner = removals | std::views::chunk_by([](const auto &a, const auto &b) { return a.Owner == b.Owner; });
        // A survivor past an owner's first erased slot moves down by the slots erased before it.
        // Every entry is found before any buffer compacts, while each slot still holds its ID.
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
            update_run(owner, false);
            RetallyMesh(r, owner);
        }
        for (const auto owner : retired) {
            update_run(owner, false);
            if (r.valid(owner)) RetallyMesh(r, owner);
        }
        slots_changed = !erased_positions.empty();
        removals.clear();
        retired.clear();
    }

    // New instances by buffer, in entity order within each buffer.
    std::vector<std::pair<state::Entity, state::Entity>> placements;
    for (auto entity : reactive(r, Change::RenderInstanceCreated)) {
        if (!r.valid(entity)) continue;
        if (const auto *ri = r.try_get<const RenderInstance>(entity); ri && ri->BufferIndex == UINT32_MAX) placements.emplace_back(ri->Entity, entity);
    }
    std::ranges::sort(placements);
    auto groups = placements | std::views::chunk_by([](const auto &a, const auto &b) { return a.first == b.first; });
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
    std::vector<uint8_t> states;
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
            // The moved slots keep their drawing-order entries, found by the IDs both ranges hold.
            const auto ids = buffers.Instances.ObjectIdBuffer.GetSpan<uint32_t>();
            const auto list = buffers.GpuInstanceSlots.GetMutableSpan<uint32_t>();
            for (uint32_t i = 0u; i < mb.InstanceCount; ++i) list[SlotPosition(list, ids, old_range.Offset + i)] = mb.InstanceRange.Offset + i;
            RebaseRenderInstances(r, ids, {mb.InstanceRange.Offset, mb.InstanceCount});
            buffers.Instances.Free(old_range);
            slots_changed = true;
        }
        object_ids.resize(n);
        states.resize(n);
        instance_records.assign(n, {});
        const auto base_index = mb.InstanceRange.Offset + mb.InstanceCount;
        const auto *mesh_buffers = TryRecordOf(r, buffer_entity);
        const auto mesh_record = mesh_buffers && mesh_buffers->RenderTopology != InvalidOffset ? mesh_buffers->StoreId : InvalidOffset;
        for (uint32_t j = 0; j < n; ++j) {
            const auto instance_entity = group[j].second;
            r.edit<RenderInstance>(instance_entity).BufferIndex = base_index + j;
            inserted_slots.push_back(base_index + j);
            object_ids[j] = ObjectId(instance_entity);
            states[j] = InstanceStateBits(r, instance_entity);
            instance_records[j] = {
                .Mesh = mesh_record,
                .ObjectId = ObjectId(instance_entity),
            };
        }
        // Transform slots stay unwritten here, and the world-transform recompute and the bone display pass write them before submit.
        buffers.Instances.ObjectIdBuffer.Update(as_bytes(object_ids), uint64_t(base_index) * sizeof(uint32_t));
        buffers.Instances.StateBuffer.Update(as_bytes(states), uint64_t(base_index) * sizeof(uint8_t));
        buffers.Instances.RecordBuffer.Update(as_bytes(instance_records), uint64_t(base_index) * sizeof(InstanceRecord));
        // Bounds reduction populates mesh instances; extras retain empty bounds and bypass culling.
        std::ranges::fill(buffers.Instances.GetMutableBounds({base_index, n}), AABB{});
        mb.InstanceCount = new_total;
        for (const auto &[_, instance_entity] : group) newly_inserted.push_back(instance_entity);
        update_run(buffer_entity, true);
        RetallyMesh(r, buffer_entity);
    }
    if (!inserted_slots.empty()) MergeInsertedSlots(buffers, std::move(inserted_slots));
    slots_changed |= !newly_inserted.empty();
    if (slots_changed) scene.OverlayJobsDirty = true;
    return {std::move(newly_inserted), std::move(new_mesh_entities), std::move(new_extras_entities), slots_changed, layout};
}

// Resize viewport GPU resources and return whether their extent changed.
bool SyncViewportRenderResources(state::Scene &r, state::Entity viewport) {
    auto &targets = r.Context.get<RenderTargets>();
    const auto render_extent_px = RenderExtentPx(r);
    const auto render_extent = std::bit_cast<mtl::Extent2D>(render_extent_px);
    if (render_extent.Width == 0 || render_extent.Height == 0) return false;
    if (targets.BuiltColorExtent() == render_extent) return false;

    const auto &ctx = r.Context.get<const mtl::Context>();
    const auto &samplers = r.Context.get<const RenderSamplerSlots>();
    auto &slots = r.Context.get<mtl::BindlessSet>();
    // Wait for the live consumer (ImGui) to finish sampling the old resources before recreating them.
    if (auto *consumer = r.Context.get<const ViewportConsumerFence>().Value) {
        const mtl::AutoreleaseScope native_scope;
        consumer->waitUntilCompleted();
    }
    targets.SetExtent(ctx, render_extent, slots);
    {
        const auto shading = r.get<const ViewportDisplay>(viewport).ViewportShading;
        const bool is_pbr = !WorkbenchShading(shading);
        const bool want_transmission = is_pbr && GetActivePbrLighting(r, viewport, shading).RealTransmission && GetPipelines(r).Main.Compiler.HasFeature(PbrFeature::Transmission);
        targets.EnsureTransmissionResources(ctx, render_extent, want_transmission);
    }
    {
        const profile::CpuScope scope{"UpdateSamplerSlots"};
        const auto set_sampler = [&](uint32_t slot, SampledTexture sampled) { slots.SetSampler({SlotType::Sampler, slot}, sampled.Texture, sampled.Sampler); };
        set_sampler(samplers.Silhouette, targets.Nearest(&targets.Resources->SilhouetteImage));
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
