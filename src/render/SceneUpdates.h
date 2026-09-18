#pragma once

#include "mesh/ElementIndicesGpu.h"
#include "state/Entity.h"
#include <span>
#include <vector>
struct Mesh;
struct GpuBuffers;
struct MeshBuffers;
struct MeshStore;

struct SyncResult {
    std::vector<state::Entity> NewlyInserted;
    std::vector<state::Entity> NewMeshEntities;
    std::vector<state::Entity> NewExtrasEntities;
    bool Compacted{false};
};

void RepointMeshInstances(state::Scene &, std::span<const state::Entity>);
void BuildMeshletsNow(state::Scene &, std::span<const state::Entity>);
void BuildBoneMeshletsNow(state::Scene &, std::span<const state::Entity>);
bool DrawsElementIndices(const state::Scene &, state::Entity);
bool DrawsStoredCorners(const Mesh &);
bool NeedsElementIndices(const Mesh &, bool);
// Allocates a mesh's missing draw indices: fan triangles for an n-gon mesh, and edges and vertices when `overlay_indices` asks for them or the mesh has no faces.
// A face mesh's triangles and edge endpoints queue one GPU work item, and an edge mesh writes its own edges.
void WriteElementIndices(GpuBuffers &, const MeshStore &, const Mesh &, MeshBuffers &, bool overlay_indices, std::vector<ElementIndicesWork> &pending);
SyncResult SyncModelsBuffers(state::Scene &);
bool SyncViewportRenderResources(state::Scene &, state::Entity);
uint8_t InstanceStateBits(const state::Scene &, state::Entity);
