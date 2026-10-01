#pragma once

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

// Returns whether an instance started or stopped drawing meshlets.
bool RepointMeshInstances(state::Scene &, std::span<const state::Entity>);
// Repoints meshes whose render data changed in place, and returns whether the scene structure changed with them.
// It changes when such a mesh is posed, draws a topology the pipelines lack, starts or stops drawing, or moves an edit binding its instance records hold.
bool RepointChangedMeshes(state::Scene &, std::span<const state::Entity>);
// Builds and places level-zero meshlets for the meshes, in input order.
void BuildMeshletsNow(state::Scene &, std::span<const state::Entity>);
// Refreshes primitive dependencies and invalidates coarse groups when those dependencies change.
void RefreshClusterLodAttributes(state::Scene &, std::span<const state::Entity>);
// Every instance of an edited mesh draws original geometry, since an element pick can land on any of them.
bool EditPinsFinest(const selection::PrimaryEditInstanceMap &, const GpuSceneState &, state::Entity mesh_entity);
// Builds the cluster hierarchy for each face mesh that lacks one and that an unpinned instance draws, and returns whether any mesh took one.
bool BuildDemandedClusterLods(state::Scene &, bool edit_mode);
void BuildBoneMeshletsNow(state::Scene &, std::span<const state::Entity>);
// Borrows a dense canonical face corner set as the mesh's triangle indices.
void AssignFaceIndices(const MeshStore &, const Mesh &, MeshBuffers &);
SyncResult SyncModelsBuffers(state::Scene &);
bool SyncViewportRenderResources(state::Scene &, state::Entity);
uint8_t InstanceStateBits(const state::Scene &, state::Entity);
