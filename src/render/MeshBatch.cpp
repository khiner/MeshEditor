#include "render/MeshBatch.h"

#include "Parallel.h"
#include "Profile.h"
#include "render/MeshConnectivityGpu.h"
#include "render/VertexWeldGpu.h"

#include <entt/entity/registry.hpp>

std::vector<CreatedMesh> CreateMeshes(entt::registry &r, std::span<MeshSource> sources) {
    auto &meshes = r.ctx().get<MeshStore>();
    // One reserve per arena for the whole batch, so no allocation below grows a buffer.
    for (const auto &source : sources) {
        meshes.PlanCreate(source.Data, source.Primitives, source.Deform.has_value(), source.Morph ? source.Morph->TargetCount : 0u, source.Attrs);
    }
    meshes.CommitReserves();

    std::vector<PreparedMesh> prepared(sources.size());
    {
        const profile::CpuScope scope{"PrepareMeshes"};
        ParallelFor(uint32_t(sources.size()), [&](uint32_t i) {
            prepared[i] = PrepareMeshSources(sources[i].Data, sources[i].Attrs, sources[i].Primitives);
        });
    }
    // The host position and corner copies are released, so nothing reads what the weld is about to change.
    std::vector<uint32_t> ids(sources.size());
    {
        const profile::CpuScope scope{"CreateMeshSource"};
        for (uint32_t i = 0; i < sources.size(); ++i) {
            auto &source = sources[i];
            ids[i] = meshes.CreateMeshSource(source.Data);
            meshes.CreateDeformSource(ids[i], source.Deform, source.Morph);
            // The tangent deltas are the one vertex-domain channel no arena holds, so the weld reads
            // them here and hands them back compacted.
            if (source.Morph) prepared[i].MorphTangentDeltas = std::move(source.Morph->TangentDeltas);
            source.Data.Positions = std::vector<vec3>{};
            if (source.Data.FaceCount() > 0) source.Data.FaceCorners = std::vector<uint32_t>{};
            source.Deform.reset();
            source.Morph.reset();
        }
    }
    {
        // Only a source that asks welds, and every mesh's connectivity reads the corners the weld rewrote.
        std::vector<WeldTarget> weld_targets;
        weld_targets.reserve(sources.size());
        for (uint32_t i = 0; i < sources.size(); ++i) {
            if (sources[i].Weld) weld_targets.emplace_back(ids[i], &sources[i].Data, &prepared[i]);
        }
        WeldMeshesNow(r, weld_targets);
    }
    {
        // The storage each build fills is allocated in source order, and the builds then run at once.
        const profile::CpuScope scope{"BuildConnectivity"};
        for (uint32_t i = 0; i < sources.size(); ++i) {
            const auto &data = sources[i].Data;
            const uint32_t halfedges = data.FaceCount() > 0 ? uint32_t(meshes.GetFaceCornerRange(ids[i]).Count) : uint32_t(data.Edges.size()) * 2u;
            const bool triangles = data.FaceCount() > 0 && halfedges == 3 * data.FaceCount();
            meshes.AllocateConnectivity(
                ids[i], meshes.GetVerticesRange(ids[i]).Count, halfedges, data.FaceCount(), !triangles && data.FaceCount() > 0
            );
        }
        std::vector<ConnectivityTarget> targets;
        targets.reserve(sources.size());
        for (uint32_t i = 0; i < sources.size(); ++i) targets.emplace_back(ids[i], &sources[i].Data);
        const auto host_targets = BuildConnectivityNow(r, targets);
        std::vector<BuiltConnectivity> built(host_targets.size());
        ParallelFor(uint32_t(host_targets.size()), [&](uint32_t i) {
            built[i] = BuildPreparedConnectivity(meshes, host_targets[i].StoreId, *host_targets[i].Data, meshes.GetConnectivityStorage(host_targets[i].StoreId));
        });
        for (uint32_t i = 0; i < host_targets.size(); ++i) meshes.PlaceConnectivity(host_targets[i].StoreId, built[i]);
    }

    std::vector<CreatedMesh> created;
    created.reserve(sources.size());
    for (uint32_t i = 0; i < sources.size(); ++i) {
        auto &source = sources[i];
        created.emplace_back(meshes.CreateMesh(
            ids[i], std::move(source.Data), std::move(source.Attrs), std::move(source.Primitives), std::move(prepared[i]), source.FlatShaded
        ));
    }
    return created;
}

CreatedMesh CreateMesh(entt::registry &r, MeshSource source) { return std::move(CreateMeshes(r, {&source, 1}).front()); }
