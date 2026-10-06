#pragma once
#include "mesh/MeshStore.h"

#include "selection/Selection.h"
#include "state/Entity.h"
#include <span>
#include <vector>
struct GpuBuffers;
struct GpuSceneState;
namespace mtl {
struct ComputeChain;
}

struct SyncResult {
    std::vector<state::Entity> NewlyInserted;
    std::vector<state::Entity> NewMeshEntities;
    std::vector<state::Entity> NewExtrasEntities;
    bool SlotsChanged{false}; // An instance slot was inserted, removed or moved.
    // A mesh gained its first instances, or a mesh with per-instance deformation or a new drawn topology changed its instances, so the scene layout rebuilds.
    bool LayoutChanged{false};
};

// Moves the mesh's contribution to the flag totals to its record's flags, visible instance count and meshlet work.
void RetallyMesh(state::Scene &, state::Entity mesh_entity);
// Points the meshes' instances at their current render records and refreshes their meshlet work.
// Each mesh's record display fields rederive at the end of the settle pass.
void RepointMeshInstances(state::Scene &, std::span<const state::Entity>);
// Repoints meshes whose render data changed in place, and returns whether the scene layout changed with them.
// It changes when such a mesh is posed or draws a topology the pipelines lack.
bool RepointChangedMeshes(state::Scene &, std::span<const state::Entity>);
// Builds and places level-zero meshlets for the meshes and the procedural bone meshes in one batch on the chain, in input order.
// The build submits the chain for its counts and publication, and the refreshed LOD attributes run with its next submit.
void BuildMeshlets(state::Scene &, mtl::ComputeChain &, std::span<const state::Entity> mesh_entities, std::span<const state::Entity> bone_entities);
// Recomputes each material's required LOD attributes and returns whether any differ from the attributes the primitives hold.
bool RefreshMaterialLodAttributes(state::Scene &);
// Refreshes primitive dependencies and records the refit of coarse groups those changes invalidate.
void RefreshClusterLodAttributes(state::Scene &, mtl::ComputeChain &, std::span<const state::Entity>);
// Every instance of an edited mesh draws original geometry, since an element pick can land on any of them.
bool EditPinsFinest(const selection::PrimaryEditInstanceMap &, const GpuSceneState &, state::Entity mesh_entity);
// Builds the cluster hierarchy for each face mesh that lacks one and that an unpinned instance draws, and returns whether any mesh took one.
bool BuildDemandedClusterLods(state::Scene &, state::Entity viewport);
// Derives a RenderInstance for the Instance of each entity in Change::InstanceVisibility, and writes its Hidden bit into a placed instance's state.
// Returns whether a placed instance's visibility changed.
bool DeriveRenderInstances(state::Scene &);
SyncResult SyncModelsBuffers(state::Scene &);
bool SyncViewportRenderResources(state::Scene &, state::Entity);
uint8_t InstanceStateBits(const state::Scene &, state::Entity);
// Whether a mesh's selected instances outline through the screen-space silhouette, which face meshes do and extras, bones, and joints do not.
bool IsSilhouetteEligible(const state::Scene &, state::Entity mesh_entity);
