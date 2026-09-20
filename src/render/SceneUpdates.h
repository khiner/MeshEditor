#pragma once

#include "mesh/ElementIndicesGpu.h"
#include "selection/Selection.h"
#include "state/Entity.h"
#include <span>
#include <vector>
struct Mesh;
struct GpuBuffers;
struct GpuSceneState;
struct MeshBuffers;
struct MeshStore;

struct SyncResult {
    std::vector<state::Entity> NewlyInserted;
    std::vector<state::Entity> NewMeshEntities;
    std::vector<state::Entity> NewExtrasEntities;
    bool Compacted{false};
};

void RepointMeshInstances(state::Scene &, std::span<const state::Entity>);
// Builds and places level-zero meshlets for the meshes, in input order.
void BuildMeshletsNow(state::Scene &, std::span<const state::Entity>);
// Every instance of an edited mesh draws original geometry, since an element pick can land on any of them.
bool EditPinsFinest(const selection::PrimaryEditInstanceMap &, const GpuSceneState &, state::Entity mesh_entity);
// Builds the cluster hierarchy for each face mesh that lacks one and that an unpinned instance draws, and returns whether any mesh took one.
bool BuildDemandedClusterLods(state::Scene &, bool edit_mode);
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
