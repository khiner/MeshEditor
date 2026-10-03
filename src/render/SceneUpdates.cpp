#include "render/SceneUpdates.h"
#include "Parallel.h"
#include "Profile.h"
#include "armature/ArmatureComponents.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "metal/AutoreleaseScope.h"
#include "object/PendingSync.h"
#include "render/GpuBufferOps.h"
#include "render/GpuBuffers.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/MeshBuffers.h"
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
#include "state/Scene.h"
#include "viewport/RenderExtent.h"
#include "viewport/ViewportConsumerFence.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportRenderGpu.h"

using state::Change;
uint8_t InstanceStateBits(const state::Scene &r, state::Entity e) {
    return (r.all_of<Selected>(e) ? ElementStateSelected : 0) | (r.all_of<Active>(e) ? ElementStateActive : 0);
}

namespace {
// Refreshes an instance's meshlet counts from its mesh's counts, in the scene totals and in the totals of the counted flags its record carries.
// Returns whether the instance started or stopped drawing meshlets.
bool UpdateMeshletInstance(GpuBuffers &buffers, std::span<const InstanceRecord> records, RenderInstance &instance, uint32_t lod_node_count, uint32_t meshlet_count) {
    const auto flags = instance.BufferIndex < records.size() ? records[instance.BufferIndex].Flags & GpuBuffers::CountedMeshletFlags : 0u;
    const auto tally = [&](bool add) {
        const auto apply = [&](uint64_t &total, uint64_t value) { total = add ? total + value : total - value; };
        apply(buffers.LodNodeCount, instance.LodNodeCount);
        apply(buffers.MeshletInstanceCount, instance.MeshletCount);
        if (!instance.MeshletCount) return;
        for (auto bits = flags; bits; bits &= bits - 1u) {
            auto &work = buffers.FlagWork(bits & (~bits + 1u));
            apply(work.Nodes, instance.LodNodeCount);
            apply(work.Meshlets, instance.MeshletCount);
        }
    };
    const bool drawing = instance.MeshletCount > 0u;
    tally(false);
    instance.LodNodeCount = lod_node_count;
    instance.MeshletCount = meshlet_count;
    tally(true);
    return drawing != (instance.MeshletCount > 0u);
}
} // namespace

// Assign placed primitives to the affected meshes' instance ranges.
bool RepointMeshInstances(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    auto &buffers = r.Context.get<GpuBuffers>();
    bool drawing_changed = false;
    for (const auto mesh_entity : mesh_entities) {
        const auto *models = r.try_get<const ModelsBuffer>(mesh_entity);
        if (!models || !models->InstanceCount) continue;
        const auto first = models->InstanceRange.Offset;
        const auto ids = buffers.Instances.ObjectIdBuffer.GetSpan<uint32_t>({first,models->InstanceCount});
        auto records = buffers.Instances.RecordBuffer.GetMutableSpan<InstanceRecord>({first,models->InstanceCount});
        const auto all_records = buffers.Instances.RecordBuffer.GetSpan<InstanceRecord>();
        const auto &mesh_buffers = MeshBuffersOf(r,mesh_entity);
        const auto primitive_count = buffers.PrimitiveCount(mesh_buffers);
        const auto lod_node_count = buffers.ActiveMeshlets.Count(mesh_buffers.NodeRoot);
        const auto meshlet_count = buffers.MeshletCount(mesh_buffers);
        for (uint32_t i=0u; i<ids.size(); ++i) {
            if (!ids[i]) continue;
            const auto instance_entity = r.EntityAt(ids[i]-1u);
            if (instance_entity==state::Null) continue;
            const auto *ri = r.try_get<const RenderInstance>(instance_entity);
            if (!ri || ri->Entity!=mesh_entity || ri->BufferIndex!=first+i) continue;
            auto &record = records[i];
            record.PrimitiveRoot = mesh_buffers.PrimitiveRoot;
            record.PrimitiveCount = primitive_count;
            record.Mesh = OffsetOrInvalid(mesh_buffers.MeshRecord);
            drawing_changed = UpdateMeshletInstance(buffers, all_records, r.edit<RenderInstance>(instance_entity), lod_node_count, meshlet_count) || drawing_changed;
        }
    }
    return drawing_changed;
}

bool RepointChangedMeshes(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    auto &buffers = r.Context.get<GpuBuffers>();
    auto &scene = r.Context.get<GpuSceneState>();
    const auto &meshes = r.Context.get<const MeshStore>();
    bool changed = RepointMeshInstances(r, mesh_entities);
    const EditSelectionStorage unbound{};
    for (const auto entity : mesh_entities) {
        const auto *models = r.try_get<const ModelsBuffer>(entity);
        if (!models || !models->InstanceCount) continue;
        const auto &owner = MeshBuffersOf(r, entity);
        const auto work = scene.EditWork.find(entity);
        changed = changed || scene.PosedByEntity.contains(entity) || (work != scene.EditWork.end() && work->second.RequiresPose) ||
            owner.RenderTopology >= 32u || !(buffers.MeshletTopologyMask & (1u << owner.RenderTopology));
        const auto &record = meshes.Get(owner.StoreId);
        const auto storage = meshes.GetEditSelectionStorage(owner.StoreId);
        const auto edge_offset = meshes.Arenas().EdgeHalfedges.First(record.EdgeData);
        for (const auto &instance : buffers.Instances.RecordBuffer.GetSpan<InstanceRecord>({models->InstanceRange.Offset, models->InstanceCount})) {
            const bool bound = std::memcmp(&instance.Selection, &unbound, sizeof(unbound)) != 0;
            changed = changed || (bound && std::memcmp(&instance.Selection, &storage, sizeof(storage)) != 0) ||
                (instance.EditEdgeSharpnessOffset != InvalidOffset && instance.EditEdgeSharpnessOffset != edge_offset);
        }
        if (scene.MeshletEditOverlayMeshes.contains(entity)) scene.MeshletEditHasSharpEdges |= meshes.GetEdgeSharpnessSummary(owner.StoreId).Any;
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
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto materials = GetMaterials(r);
    const auto debug = LodDebugChannel(r);
    for (const auto entity : mesh_entities) {
        auto *owner = TryMeshBuffers(r, entity);
        if (!owner) continue;
        if (!ClusterLodApplies(owner->RenderTopology == uint32_t(MeshPrimitiveTopology::Triangle), owner->Level0Count)) continue;
        const auto &record = meshes.Get(owner->StoreId);
        const auto assignments = meshes.Arenas().PrimitiveMaterials.Get(record.PrimitiveMaterials);
        const bool authored_tangents = (record.CornerAttributes & MeshAttributeBit_Tangent) != 0u;
        const bool coarse = buffers.ClusterGroupCount(*owner) != 0u;
        std::vector<uint32_t> changed;
        buffers.ForEachPrimitive(*owner, [&](uint32_t id, const PrimitiveRecord &primitive) {
            const uint32_t material = primitive.PrimitiveIndex < assignments.size() ? assignments[primitive.PrimitiveIndex] : 0u;
            const uint32_t required = MaterialLodAttributes(materials[material], debug, authored_tangents);
            if (required == primitive.LodAttributes) return;
            if (coarse) changed.push_back(id);
            buffers.Primitives.GetMutable({id, 1u})[0].LodAttributes = required;
        });
        if (changed.empty()) continue;
        std::ranges::sort(changed);
        std::vector<uint32_t> groups;
        buffers.ActiveMeshlets.ForEach(owner->GroupRoot, [&](uint32_t group) {
            const auto &links = buffers.GroupLinks.Get({group, 1u})[0];
            if (!links.MemberCount) return;
            const auto member = buffers.GroupClusterIds.Get({links.MemberOffset, 1u})[0];
            const auto primitive = buffers.Meshlets.Get({member, 1u})[0].Primitive;
            if (std::ranges::binary_search(changed, primitive)) groups.push_back(group);
        });
        auto touched = InvalidateClusterGroups(r, entity, groups);
        if (touched.empty()) continue;
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
        EditLodNodes(r, chain, MeshBuffersOf(r, entity), {}, {}, touched);
    }
}

void BuildMeshletsNow(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    if (mesh_entities.empty()) return;
    for (auto e : mesh_entities) ReleaseMeshEditWork(r, e);
    const profile::CpuScope scope{"BuildMeshlets"};
    auto &buffers = r.Context.get<GpuBuffers>();
    buffers.PreludeStale = true;
    const auto &meshes = r.Context.get<const MeshStore>();
    std::vector<MeshletBuildSource> sources;
    sources.reserve(mesh_entities.size());
    for (const auto entity : mesh_entities) {
        const auto mesh = GetMesh(r, entity);
        const auto topology = mesh.PrimitiveTopology();
        const bool faces = topology == 0u, lines = topology == 1u;
        auto &mb=buffers.MeshOf(mesh.GetStoreId());
        // A record changing topology releases its former render ranges.
        if (mb.RenderTopology!=InvalidOffset && mb.RenderTopology!=topology) {
            auto vertices=mb.Vertices;
            vertices.Count=mesh.VertexCount();
            buffers.Release(mb);
            mb=MeshBuffers{.Vertices=vertices};
        }
        MeshletBuildSource source{
            .Destination = &mb,
            .Mesh = BuildMeshRecord(buffers, mb, meshes, mesh.GetStoreId(), faces, lines), .StoreId = mesh.GetStoreId(),
            .Topology = topology,
            .ElementCount = faces ? mesh.TriangleIndexCount() / 3u : lines ? mesh.EdgeCount() : mesh.VertexCount(),
        };
        sources.push_back(std::move(source));
    }
    mtl::ComputeChain chain{buffers.Ctx};
    BuildGpuMeshlets(r, chain, sources);
    RefreshClusterLodAttributes(r, chain, mesh_entities);
    chain.Submit();
    RepointMeshInstances(r, mesh_entities);
    r.Context.get<GpuSceneState>().LodDemand.insert(mesh_entities.begin(), mesh_entities.end());
}

bool EditPinsFinest(const selection::PrimaryEditInstanceMap &primaries, const GpuSceneState &scene, state::Entity mesh_entity) {
    return primaries.contains(mesh_entity) || scene.EditWork.contains(mesh_entity);
}

bool BuildDemandedClusterLods(state::Scene &r, bool edit_mode) {
    auto &scene = r.Context.get<GpuSceneState>();
    if (scene.LodDemand.empty()) return false;
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto &meshes = r.Context.get<const MeshStore>();
    // A mesh leaves the demand set once it has a hierarchy or its meshlets take none.
    std::erase_if(scene.LodDemand, [&](state::Entity entity) {
        const auto *handle = r.valid(entity) ? r.try_get<const MeshHandle>(entity) : nullptr;
        const auto *mb = handle ? buffers.TryMeshOf(handle->StoreId) : nullptr;
        return !mb || buffers.ClusterGroupCount(*mb) != 0u || !ClusterLodApplies(Mesh{meshes, handle->StoreId}.FaceCount() > 0u, buffers.MeshletCount(*mb));
    });
    std::vector<state::Entity> demanded{scene.LodDemand.begin(), scene.LodDemand.end()};
    if (edit_mode && !demanded.empty()) {
        const auto primaries = selection::ComputePrimaryEditInstances(r);
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
        const auto &mb = MeshBuffersOf(r,entity);
        const auto mesh=GetMesh(r,entity);
        // After an edit, the build covers exactly the triangles the live finest clusters hold, per primitive in walk order.
        if (mb.MeshletRevision>1u) {
            auto &triangles=live_triangles[i];
            auto &primitive_counts=live_primitive_counts[i];
            buffers.ForEachPrimitive(mb,[&](uint32_t, const PrimitiveRecord &primitive) {
                const auto root=primitive.LodFinestNode==InvalidOffset ? InvalidOffset :
                    buffers.LodNodes.Get({primitive.LodFinestNode,1u})[0].MeshletRoot;
                const auto start=triangles.size();
                buffers.ActiveMeshlets.ForEach(root,[&](uint32_t cluster) {
                    const auto &record=buffers.Meshlets.Get({cluster,1u})[0];
                    const auto ids=buffers.MeshletTriangleIds.Get({record.TriangleOffset,record.TriangleCount});
                    triangles.insert(triangles.end(),ids.begin(),ids.end());
                });
                primitive_counts.push_back(uint32_t(triangles.size()-start));
            });
            inputs.push_back(CaptureMeshletInputs(mesh,meshes,
                {meshes.Arenas().Triangles.Buffer.GetSpan<uvec3>(),triangles}));
        } else inputs.push_back(CaptureMeshletInputs(mesh,meshes,
            {meshes.Arenas().Triangles.Buffer.GetSpan<uvec3>(),buffers.MeshletTriangleIds.Get(mb.MeshletTriangles)}));
    }
    std::vector<ClusterLodBuild> lods(count);
    ParallelFor(count, [&](uint32_t i) { lods[i] = BuildMeshletClusterLod(buffers, MeshBuffersOf(r, demanded[i]), inputs[i],live_primitive_counts[i]); });
    for (uint32_t i = 0; i < count; ++i) {
        auto &mb = MeshBuffersOf(r,demanded[i]);
        CommitClusterLod(r,mb,lods[i]);
        // Finest IDs remain stable, so their canonical owner entries do too.
    }
    RepointMeshInstances(r, demanded);
    buffers.PreludeStale = true;
    return true;
}

// Populate standard meshlet geometry so procedural bone shaders share bounds, culling, routing, and indirect dispatch.
void BuildBoneMeshletsNow(state::Scene &r, std::span<const state::Entity> entities) {
    auto &buffers = r.Context.get<GpuBuffers>();
    std::vector<MeshletBuildSource> sources;
    sources.reserve(entities.size());
    for (const auto entity : entities) {
        auto &mb = MeshBuffersOf(r, entity);
        if (mb.FaceIndices.Count == 0u) continue;
        const uint32_t triangle_count = mb.FaceIndices.Count / 3u;
        sources.push_back({
            .Destination = &mb,
            .Mesh = {
                .VertexSlot = mb.Vertices.Slot,
                .IndexSlotOffset = mb.FaceIndices,
                .ModelSlot = buffers.Instances.TransformBuffer.Slot,
                .VertexCountOrHeadImageSlot = mb.Vertices.Count,
                .InstanceStateSlot = buffers.Instances.StateBuffer.Slot,
                .VertexOffset = mb.Vertices.Offset,
            },
            .AuxIndices = mb.EdgeIndices,
            .Topology = 0u,
            .ElementCount = triangle_count,
        });
    }
    mtl::ComputeChain chain{buffers.Ctx};
    BuildGpuMeshlets(r, chain, sources);
    chain.Submit();
    RepointMeshInstances(r, entities);
}

void AssignFaceIndices(const MeshStore &meshes, const Mesh &mesh, MeshBuffers &mb) {
    if (mesh.TriangleIndexCount() == 0 || mb.FaceIndices.Count > 0) return;
    const auto &corners=meshes.Arenas().FaceCorners;
    const auto set=meshes.Get(mesh.GetStoreId()).FaceCorners;
    if (corners.Set(set).Flags & 1u) mb.FaceIndices=corners.Slotted(set);
}

void DeriveRenderInstances(state::Scene &r) {
    for (const auto e : reactive(r, state::Change::InstanceVisibility).Entities) {
        if (!r.valid(e)) continue;
        const auto *instance = r.try_get<const Instance>(e);
        const bool visible = instance && !r.all_of<Hidden>(e);
        if (const auto *render = r.try_get<const RenderInstance>(e)) {
            if (visible && render->Entity == instance->Entity) continue;
            r.remove<RenderInstance>(e);
        }
        if (visible) r.emplace<RenderInstance>(e, instance->Entity, UINT32_MAX);
    }
}

SyncResult SyncModelsBuffers(state::Scene &r) {
    const profile::CpuScope scope{"SyncModelsBuffers"};
    auto &buffers = r.Context.get<GpuBuffers>();
    auto &meshes = r.Context.get<MeshStore>();
    // Released and restored records drop their render data ahead of this pass's builds.
    buffers.ReleaseMeshes(meshes.TakeRenderStale());
    std::vector<state::Entity> new_mesh_entities, new_extras_entities;
    for (auto e : reactive(r, Change::NewBufferEntity)) {
        if (!r.valid(e)) continue;
        // A record keeps its render data across handle changes that leave it intact.
        const auto id = DrawnStoreId(r, e);
        if (!id || buffers.TryMeshOf(*id)) continue;
        const auto &vertices = meshes.Arenas().Vertices;
        const auto set = meshes.Get(*id).Vertices;
        buffers.EmplaceMesh(*id, {{vertices.First(set), vertices.Count(set)}, vertices.Buffer.Slot});
        if (HasMesh(r, e)) new_mesh_entities.emplace_back(e);
        else if (r.all_of<ObjectExtrasTag>(e) || r.all_of<ArmatureObject>(e) || r.all_of<BoneJoint>(e)) new_extras_entities.emplace_back(e);
    }

    bool compacted = false;
    if (auto *pending = r.Context.find<PendingHide>()) {
        auto &removals = pending->Instances;
        auto &retired = pending->Retired;
        if (!std::ranges::is_sorted(retired)) std::ranges::sort(retired);
        std::erase_if(removals, [&](const auto &removal) {
            return !r.valid(removal.Owner) || std::ranges::binary_search(retired, removal.Owner) || !r.all_of<ModelsBuffer>(removal.Owner);
        });
        if (!std::ranges::is_sorted(removals)) std::ranges::sort(removals);
        removals.erase(std::unique(removals.begin(), removals.end()), removals.end());
        for (const auto erased : removals | std::views::chunk_by([](const auto &a, const auto &b) { return a.Owner == b.Owner; })) {
            auto &mb = r.edit<ModelsBuffer>(erased.front().Owner);
            buffers.Instances.CompactErase({mb.InstanceRange.Offset, mb.InstanceCount}, erased, &PendingHide::Removal::Index);
            mb.InstanceCount -= erased.size();
        }
        compacted = !removals.empty();
        if (compacted) for (const auto [entity, ri] : r.view<const RenderInstance>().each()) {
            if (ri.BufferIndex == UINT32_MAX) continue;
            const auto erased = std::ranges::equal_range(removals, ri.Entity, {}, &PendingHide::Removal::Owner);
            const auto shift = uint32_t(std::ranges::lower_bound(erased, ri.BufferIndex, {}, &PendingHide::Removal::Index) - erased.begin());
            if (shift) r.edit<RenderInstance>(entity).BufferIndex -= shift;
        }
        removals.clear();
        retired.clear();
    }

    // Return inserted instances so callers can write WorldTransform before submission.
    std::vector<state::Entity> newly_inserted;
    std::unordered_map<state::Entity, std::vector<state::Entity>> shows_by_buffer;
    for (auto entity : reactive(r, Change::RenderInstanceCreated)) {
        if (!r.valid(entity) || !r.all_of<RenderInstance>(entity)) continue;

        const auto &ri = r.get<const RenderInstance>(entity);
        if (ri.BufferIndex == UINT32_MAX) shows_by_buffer[ri.Entity].emplace_back(entity);
    }
    // Reserve all new instance slots in one allocation.
    {
        uint32_t total_new_instances = 0;
        for (const auto &[_, entities] : shows_by_buffer) total_new_instances += entities.size();
        if (total_new_instances > 0) buffers.Instances.ReserveAdditional(total_new_instances);
    }
    // Each grown buffer's prior range, re-based onto its new range in one pass after the inserts.
    struct Regrowth {
        state::Entity Buffer;
        uint32_t OldOffset, Count, NewOffset;
    };
    std::vector<Regrowth> regrowths;
    std::vector<uint32_t> object_ids;
    std::vector<uint8_t> states;
    std::vector<InstanceRecord> instance_records;
    for (auto &[buffer_entity, entities] : shows_by_buffer) {
        const uint32_t n = entities.size();
        // Defer ModelsBuffer creation until its initial capacity is known.
        if (!r.all_of<ModelsBuffer>(buffer_entity)) {
            r.emplace<ModelsBuffer>(buffer_entity, ModelsBuffer{buffers.Instances.Allocate(n), 0});
        }
        auto &mb = r.edit<ModelsBuffer>(buffer_entity);
        const auto new_total = mb.InstanceCount + n;
        if (new_total > mb.InstanceRange.Count) {
            auto old_range = mb.InstanceRange;
            const auto new_capacity = std::max(mb.InstanceRange.Count * 2, new_total);
            mb.InstanceRange = buffers.Instances.Allocate(new_capacity);
            buffers.Instances.CopyInstances(old_range.Offset, mb.InstanceRange.Offset, mb.InstanceCount);
            regrowths.push_back({buffer_entity, old_range.Offset, mb.InstanceCount, mb.InstanceRange.Offset});
            buffers.Instances.Free(old_range);
        }
        object_ids.resize(n);
        states.resize(n);
        instance_records.assign(n, {});
        const auto base_index = mb.InstanceRange.Offset + mb.InstanceCount;
        const auto *mesh_buffers = TryMeshBuffers(r, buffer_entity);
        for (uint32_t j = 0; j < n; ++j) {
            const auto instance_entity = entities[j];
            auto &render_instance = r.edit<RenderInstance>(instance_entity);
            render_instance.BufferIndex = base_index + j;
            object_ids[j] = ObjectId(instance_entity);
            states[j] = InstanceStateBits(r, instance_entity);
            auto &record = instance_records[j];
            record.ObjectId = ObjectId(instance_entity);
            if (mesh_buffers) {
                record.PrimitiveRoot = mesh_buffers->PrimitiveRoot;
                record.PrimitiveCount = buffers.PrimitiveCount(*mesh_buffers);
                record.Mesh = OffsetOrInvalid(mesh_buffers->MeshRecord);
            }
        }
        // WorldTransform slots stay unwritten here, and the WorldTransform reactive pass writes them before submit.
        buffers.Instances.ObjectIdBuffer.Update(as_bytes(object_ids), uint64_t(base_index) * sizeof(uint32_t));
        buffers.Instances.StateBuffer.Update(as_bytes(states), uint64_t(base_index) * sizeof(uint8_t));
        buffers.Instances.RecordBuffer.Update(as_bytes(instance_records), uint64_t(base_index) * sizeof(InstanceRecord));
        // Bounds reduction populates mesh instances; extras retain empty bounds and bypass culling.
        std::ranges::fill(buffers.Instances.GetMutableBounds({base_index, n}), AABB{});
        mb.InstanceCount = new_total;
        const auto records = buffers.Instances.RecordBuffer.GetSpan<InstanceRecord>();
        const auto lod_node_count = mesh_buffers ? buffers.ActiveMeshlets.Count(mesh_buffers->NodeRoot) : 0u;
        const auto meshlet_count = mesh_buffers ? buffers.MeshletCount(*mesh_buffers) : 0u;
        for (const auto instance_entity : entities) UpdateMeshletInstance(buffers, records, r.edit<RenderInstance>(instance_entity), lod_node_count, meshlet_count);
        newly_inserted.append_range(entities);
    }
    // Instances placed before their buffer grew move with its prior range.
    if (!regrowths.empty()) {
        std::ranges::sort(regrowths, {}, &Regrowth::Buffer);
        for (const auto [entity, ri] : r.view<const RenderInstance>().each()) {
            if (ri.BufferIndex == UINT32_MAX) continue;
            const auto it = std::ranges::lower_bound(regrowths, ri.Entity, {}, &Regrowth::Buffer);
            if (it == regrowths.end() || it->Buffer != ri.Entity || ri.BufferIndex < it->OldOffset || ri.BufferIndex - it->OldOffset >= it->Count) continue;
            r.edit<RenderInstance>(entity).BufferIndex = it->NewOffset + (ri.BufferIndex - it->OldOffset);
        }
    }
    return {std::move(newly_inserted), std::move(new_mesh_entities), std::move(new_extras_entities), compacted};
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
        const bool is_pbr = shading == ViewportShadingMode::MaterialPreview || shading == ViewportShadingMode::Rendered;
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
