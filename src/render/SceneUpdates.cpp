#include "render/SceneUpdates.h"
#include "Changes.h"
#include "Parallel.h"
#include "Profile.h"
#include "Reactive.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "object/PendingSync.h"
#include "render/GpuBufferOps.h"
#include "render/GpuBuffers.h"
#include "render/MeshBuffers.h"
#include "render/MeshletBuild.h"
#include "render/PickConstants.h"
#include "render/Pipelines.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "selection/SelectionBitset.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "viewport/FrameState.h"
#include "viewport/InteractionComponents.h"
#include "viewport/RenderExtent.h"
#include "viewport/ViewportConsumerFence.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportRenderGpu.h"
#include <entt/entity/registry.hpp>
#include <numeric>
#include <print>
using namespace he;
uint8_t InstanceStateBits(const entt::registry &r, entt::entity e) {
    return (r.all_of<Selected>(e) ? ElementStateSelected : 0) | (r.all_of<Active>(e) ? ElementStateActive : 0);
}

static void UpdateMeshletInstance(entt::registry &r, entt::entity instance_entity) {
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &instance = r.get<RenderInstance>(instance_entity);
    buffers.MeshletRangeCount -= instance.MeshletRangeCount;
    buffers.MeshletInstanceCount -= instance.MeshletCount;
    const auto *mesh_buffers = r.valid(instance.Entity) ? r.try_get<const MeshBuffers>(instance.Entity) : nullptr;
    instance.MeshletRangeCount = mesh_buffers ? mesh_buffers->Primitives.Count : 0;
    instance.MeshletCount = mesh_buffers ? mesh_buffers->Meshlets.Count : 0;
    buffers.MeshletRangeCount += instance.MeshletRangeCount;
    buffers.MeshletInstanceCount += instance.MeshletCount;
}

// Assign placed primitives to instances while preserving mesh and instance iteration order.
void RepointMeshInstances(entt::registry &r, std::span<const entt::entity> mesh_entities) {
    if (mesh_entities.empty()) return;
    auto &buffers = r.ctx().get<GpuBuffers>();
    std::vector<std::pair<entt::entity, uint32_t>> batch;
    batch.reserve(mesh_entities.size());
    for (uint32_t i = 0; i < mesh_entities.size(); ++i) batch.emplace_back(mesh_entities[i], i);
    std::ranges::sort(batch);

    std::vector<std::pair<uint32_t, entt::entity>> grouped;
    for (const auto [instance_entity, ri] : r.view<const RenderInstance>().each()) {
        if (ri.BufferIndex == UINT32_MAX) continue;
        const auto it = std::ranges::lower_bound(batch, ri.Entity, {}, &std::pair<entt::entity, uint32_t>::first);
        if (it == batch.end() || it->first != ri.Entity) continue;
        grouped.emplace_back(it->second, instance_entity);
    }
    std::ranges::stable_sort(grouped, {}, &std::pair<uint32_t, entt::entity>::first);

    for (const auto [mesh_index, instance_entity] : grouped) {
        const auto &mesh_buffers = r.get<const MeshBuffers>(mesh_entities[mesh_index]);
        const auto &ri = r.get<const RenderInstance>(instance_entity);
        auto &record = buffers.Instances.RecordBuffer.GetMutableSpan<InstanceRecord>({ri.BufferIndex, 1}).front();
        record.PrimitiveOffset = OffsetOrInvalid(mesh_buffers.Primitives);
        record.PrimitiveCount = mesh_buffers.Primitives.Count;
        UpdateMeshletInstance(r, instance_entity);
    }
}

// Build and place meshlet LOD data in input order to preserve deterministic arena and instance layouts.
void BuildMeshletsNow(entt::registry &r, std::span<const entt::entity> mesh_entities) {
    if (mesh_entities.empty()) return;
    for (auto e : mesh_entities) ReleaseMeshEditWork(r, e);
    const profile::CpuScope scope{"BuildMeshlets"};
    auto &buffers = r.ctx().get<GpuBuffers>();
    buffers.PreludeStale = true;
    const auto &meshes = r.ctx().get<const MeshStore>();
    const uint32_t count = uint32_t(mesh_entities.size());
    // Capture registry inputs before concurrent mesh builds.
    std::vector<MeshletBuildInputs> inputs;
    inputs.reserve(count);
    for (const auto entity : mesh_entities) {
        inputs.push_back(CaptureMeshletInputs(buffers, r.get<const MeshBuffers>(entity), GetMesh(r, entity), meshes));
    }
    // Build meshes concurrently from independent captured inputs.
    std::vector<MeshletBuild> builds(count);
    std::vector<ClusterLodBuild> lods(count);
    ParallelFor(count, [&](uint32_t i) {
        builds[i] = BuildMeshlets(inputs[i]);
        lods[i] = BuildMeshletClusterLod(inputs[i], builds[i]);
    });
    for (uint32_t i = 0; i < count; ++i) {
        auto &mb = r.get<MeshBuffers>(mesh_entities[i]);
        CommitMeshlets(buffers, mb, builds[i]);
        CommitClusterLod(buffers, mb, lods[i]);
    }
    RepointMeshInstances(r, mesh_entities);
    if (profile::Enabled) {
        uint32_t lod_meshes = 0, lod_levels = 0, lod_clusters = 0, lod_groups = 0;
        ClusterLodStats stats;
        ClusterLodLevelStats level_stats;
        for (const auto &lod : lods) {
            if (lod.Groups.empty()) continue;
            ++lod_meshes;
            lod_levels = std::max(lod_levels, lod.LevelCount);
            lod_clusters += uint32_t(lod.Clusters.size());
            lod_groups += uint32_t(lod.Groups.size());
            stats.TotalMs += lod.Stats.TotalMs;
            stats.WeldMs += lod.Stats.WeldMs;
            stats.Level0Ms += lod.Stats.Level0Ms;
            stats.HierarchyMs += lod.Stats.HierarchyMs;
            for (const auto &level : lod.Stats.Levels) {
                level_stats.PartitionMs += level.PartitionMs;
                level_stats.LockMs += level.LockMs;
                level_stats.MergeMs += level.MergeMs;
                level_stats.SimplifyMs += level.SimplifyMs;
                level_stats.ClusterizeMs += level.ClusterizeMs;
                level_stats.EmitMs += level.EmitMs;
            }
        }
        const double levels_ms = stats.TotalMs - stats.WeldMs - stats.Level0Ms - stats.HierarchyMs;
        std::println(
            "Cluster LOD: {} meshes, {} levels, {} coarse clusters, {} groups, {:.1f} ms of build ({:.1f} weld, {:.1f} source, {:.1f} levels, {:.1f} span trees)",
            lod_meshes, lod_levels, lod_clusters, lod_groups, stats.TotalMs,
            stats.WeldMs, stats.Level0Ms, levels_ms, stats.HierarchyMs
        );
        std::println(
            "Cluster LOD levels: {:.1f} ms partition, {:.1f} lock, {:.1f} merge; group CPU {:.1f} simplify, {:.1f} clusterize, {:.1f} emit",
            level_stats.PartitionMs, level_stats.LockMs, level_stats.MergeMs,
            level_stats.SimplifyMs, level_stats.ClusterizeMs, level_stats.EmitMs
        );
    }
}

// Populate standard meshlet geometry so procedural bone shaders share bounds, culling, routing, and indirect dispatch.
void BuildBoneMeshletsNow(entt::registry &r, std::span<const entt::entity> entities) {
    auto &buffers = r.ctx().get<GpuBuffers>();
    const auto &meshes = r.ctx().get<const MeshStore>();
    for (const auto entity : entities) {
        auto &mb = r.get<MeshBuffers>(entity);
        if (mb.FaceIndices.Count == 0u) continue;
        const auto indices = buffers.FaceIndexBuffer.Get(mb.FaceIndices);
        const auto vertices = meshes.GetVertices(r.get<const VertexStoreId>(entity).StoreId);
        const uint32_t triangle_count = uint32_t(indices.size() / 3u);
        std::vector<uint32_t> face_ids(triangle_count), element_primitives(triangle_count, 0u);
        std::iota(face_ids.begin(), face_ids.end(), 1u);
        MeshletBuildInputs inputs;
        inputs.Indices = indices;
        inputs.Vertices = vertices;
        inputs.ElementPrimitives = element_primitives;
        inputs.TriangleEditEdges.assign(indices.size(), InvalidOffset);
        inputs.PrimitiveTriangleRanges = {{0u, 0u, triangle_count}};
        inputs.Weld.TriangleFaceIds = face_ids;
        inputs.TriangleCount = triangle_count;
        inputs.FaceTopology = true;
        inputs.PrimitiveDraws.push_back({
            .VertexSlot = mb.Vertices.Slot,
            .IndexSlotOffset = mb.FaceIndices,
            .ModelSlot = buffers.Instances.TransformBuffer.Slot,
            .VertexCountOrHeadImageSlot = mb.Vertices.Count,
            .InstanceStateSlot = buffers.Instances.StateBuffer.Slot,
            .VertexOffset = mb.Vertices.Offset,
        });
        auto build = BuildMeshlets(inputs);
        assert(build.Primitives.size() == 1u);
        build.Primitives.front().AuxIndices = r.all_of<ArmatureObject>(entity) ?
            r.get<const BoneAdjacencyIndices>(entity).Indices :
            mb.EdgeIndices;
        CommitMeshlets(buffers, mb, build);
    }
    RepointMeshInstances(r, entities);
}

// Allocate edge and vertex indices on demand for overlays, wireframe shading, and non-triangle meshes.
// Vertex normal indicators require incident edge indices to determine their length.
bool DrawsElementIndices(const entt::registry &r, entt::entity viewport) {
    const auto mode = r.get<const Interaction>(viewport).Mode;
    const auto &display = r.get<const ViewportDisplay>(viewport);
    return display.ViewportShading == ViewportShadingMode::Wireframe ||
        mode == InteractionMode::Edit || mode == InteractionMode::Excite ||
        ElementMaskContains(display.NormalOverlays, Element::Vertex);
}

// Return the corner range directly for triangle-only meshes and a triangulated range for n-gons.
bool DrawsStoredCorners(const Mesh &mesh) {
    return mesh.TriangleIndexCount() > 0 && mesh.TriangleIndexCount() == mesh.CornerVertices().size();
}

// Use edge or vertex indices as geometry for meshes without faces.
bool NeedsElementIndices(const Mesh &mesh, bool overlay_indices) {
    return overlay_indices || mesh.FaceCount() == 0;
}

void WriteElementIndices(GpuBuffers &buffers, const Mesh &mesh, MeshBuffers &mb) {
    if (mesh.EdgeCount() > 0 && mb.EdgeIndices.Count == 0) {
        auto [sr, dest] = buffers.AllocateIndices(mesh.EdgeCount() * 2, IndexKind::Edge);
        mesh.WriteEdgeIndices(dest);
        mb.EdgeIndices = sr;
        // The meshlet build captured the edge indices into the mesh's primitive records, so records built before the indices existed take them now.
        for (auto &record : buffers.Primitives.Buffer.GetMutableSpan<PrimitiveRecord>(mb.Primitives)) record.AuxIndices = mb.EdgeIndices;
    }
    if (mesh.VertexCount() > 0 && mb.VertexIndices.Count == 0) {
        auto [sr, dest] = buffers.AllocateIndices(mesh.VertexCount(), IndexKind::Vertex);
        std::iota(dest.begin(), dest.end(), 0u);
        mb.VertexIndices = sr;
    }
}

SyncResult SyncModelsBuffers(entt::registry &r) {
    auto &buffers = r.ctx().get<GpuBuffers>();
    std::vector<entt::entity> new_mesh_entities, new_extras_entities;
    for (auto e : reactive<changes::NewBufferEntity>(r)) {
        if (!r.valid(e) || !r.all_of<MeshBuffers>(e)) continue;
        if (HasMesh(r, e)) new_mesh_entities.emplace_back(e);
        else if (r.all_of<ObjectExtrasTag>(e) || r.all_of<ArmatureObject>(e) || r.all_of<BoneJoint>(e)) new_extras_entities.emplace_back(e);
    }

    bool compacted = false;
    for (auto [buffer_entity, pending] : r.view<PendingHide>().each()) {
        // Erase in descending order to keep remaining batch indices stable.
        auto &indices = pending.BufferIndices;
        std::sort(indices.begin(), indices.end(), std::greater<>());
        auto &mb = r.get<ModelsBuffer>(buffer_entity);
        for (const auto global_idx : indices) {
            buffers.Instances.CompactErase(global_idx, mb.InstanceRange.Offset + mb.InstanceCount);
            --mb.InstanceCount;
        }
        compacted = true;
        for (auto [_, ri] : r.view<RenderInstance>().each()) {
            if (ri.Entity != buffer_entity || ri.BufferIndex == UINT32_MAX) continue;
            uint32_t shift = 0;
            for (const auto erased_idx : indices) {
                if (erased_idx < ri.BufferIndex) ++shift;
            }
            if (shift > 0) ri.BufferIndex -= shift;
        }
        r.remove<PendingHide>(buffer_entity);
    }

    // Return inserted instances so callers can write WorldTransform before submission.
    std::vector<entt::entity> newly_inserted;
    std::unordered_map<entt::entity, std::vector<entt::entity>> shows_by_buffer;
    for (auto entity : reactive<changes::RenderInstanceCreated>(r)) {
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
    std::vector<uint32_t> object_ids;
    std::vector<uint8_t> states;
    std::vector<InstanceRecord> instance_records;
    for (auto &[buffer_entity, entities] : shows_by_buffer) {
        const uint32_t n = entities.size();
        // Defer ModelsBuffer creation until its initial capacity is known.
        if (!r.all_of<ModelsBuffer>(buffer_entity)) {
            r.emplace<ModelsBuffer>(buffer_entity, ModelsBuffer{buffers.Instances.Allocate(n), 0});
        }
        auto &mb = r.get<ModelsBuffer>(buffer_entity);
        const auto new_total = mb.InstanceCount + n;
        if (new_total > mb.InstanceRange.Count) {
            auto old_range = mb.InstanceRange;
            const auto new_capacity = std::max(mb.InstanceRange.Count * 2, new_total);
            mb.InstanceRange = buffers.Instances.Allocate(new_capacity);
            buffers.Instances.CopyInstances(old_range.Offset, mb.InstanceRange.Offset, mb.InstanceCount);
            for (auto [other_entity, ri] : r.view<RenderInstance>().each()) {
                if (ri.Entity == buffer_entity && ri.BufferIndex != UINT32_MAX) {
                    ri.BufferIndex = mb.InstanceRange.Offset + (ri.BufferIndex - old_range.Offset);
                }
            }
            buffers.Instances.Free(old_range);
        }
        object_ids.resize(n);
        states.resize(n);
        instance_records.assign(n, {});
        const auto base_index = mb.InstanceRange.Offset + mb.InstanceCount;
        const auto *mesh_buffers = r.try_get<const MeshBuffers>(buffer_entity);
        for (uint32_t j = 0; j < n; ++j) {
            const auto instance_entity = entities[j];
            auto &render_instance = r.get<RenderInstance>(instance_entity);
            render_instance.BufferIndex = base_index + j;
            object_ids[j] = render_instance.ObjectId;
            states[j] = InstanceStateBits(r, instance_entity);
            auto &record = instance_records[j];
            record.ObjectId = render_instance.ObjectId;
            if (mesh_buffers) {
                record.PrimitiveOffset = OffsetOrInvalid(mesh_buffers->Primitives);
                record.PrimitiveCount = mesh_buffers->Primitives.Count;
            }
        }
        // WorldTransform slots stay unwritten here, and the WorldTransform reactive pass writes them before submit.
        buffers.Instances.ObjectIdBuffer.Update(as_bytes(object_ids), uint64_t(base_index) * sizeof(uint32_t));
        buffers.Instances.StateBuffer.Update(as_bytes(states), uint64_t(base_index) * sizeof(uint8_t));
        buffers.Instances.RecordBuffer.Update(as_bytes(instance_records), uint64_t(base_index) * sizeof(InstanceRecord));
        // Bounds reduction populates mesh instances; extras retain empty bounds and bypass culling.
        std::ranges::fill(buffers.Instances.GetMutableBounds({base_index, n}), AABB{});
        mb.InstanceCount = new_total;
        for (const auto instance_entity : entities) UpdateMeshletInstance(r, instance_entity);
        newly_inserted.append_range(entities);
    }
    return {std::move(newly_inserted), std::move(new_mesh_entities), std::move(new_extras_entities), compacted};
}

// Resize viewport GPU resources and return whether their extent changed.
bool SyncViewportRenderResources(entt::registry &r, entt::entity viewport) {
    auto &pipelines = r.ctx().get<Pipelines>();
    const auto render_extent_px = RenderExtentPx(r);
    const auto render_extent = std::bit_cast<mtl::Extent2D>(render_extent_px);
    if (render_extent.Width == 0 || render_extent.Height == 0) return false;
    if (pipelines.BuiltColorExtent() == render_extent) return false;

    const auto &ctx = r.ctx().get<const mtl::Context>();
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    auto &slots = r.ctx().get<mtl::BindlessSet>();
    // Wait for the live consumer (ImGui) to finish sampling the old resources before recreating them.
    if (auto *consumer = r.ctx().get<const ViewportConsumerFence>().Value) consumer->waitUntilCompleted();
    pipelines.Main.SetExtent(ctx, render_extent, slots);
    {
        const auto shading = r.get<const ViewportDisplay>(viewport).ViewportShading;
        const bool is_pbr = shading == ViewportShadingMode::MaterialPreview || shading == ViewportShadingMode::Rendered;
        const bool want_transmission = is_pbr && GetActivePbrLighting(r, viewport, shading).RealTransmission && pipelines.Main.Compiler.HasFeature(PbrFeature::Transmission);
        pipelines.Main.EnsureTransmissionResources(ctx, render_extent, want_transmission);
    }
    {
        const profile::CpuScope scope{"UpdateSelectionSlots"};
        const auto &main = pipelines.Main;
        const auto set_sampler = [&](uint32_t slot, SampledTexture sampled) { slots.SetSampler({SlotType::Sampler, slot}, sampled.Texture, sampled.Sampler); };
        set_sampler(sel_slots.SilhouetteSampler, main.Nearest(&main.Resources->SilhouetteImage));
        set_sampler(sel_slots.SceneColorSampler, main.SceneColorSampler());
        set_sampler(sel_slots.OverlayColorSampler, main.OverlayColorSampler());
        set_sampler(sel_slots.TransmissionSampler, main.TransmissionSampler());
        set_sampler(sel_slots.MotionBlurOutputSampler, main.MotionBlurOutputSampler());
        set_sampler(sel_slots.VelocitySampler, main.Nearest(nullptr));
        set_sampler(sel_slots.SceneDepthSampler, main.SceneDepthSampler());
        set_sampler(sel_slots.DepthPyramidSampler, main.DepthPyramidSampler());
    }
    return true;
}
