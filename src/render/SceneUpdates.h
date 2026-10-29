#pragma once

#include "numeric/uvec2.h"
#include "selection/Selection.h"
#include <span>
struct GpuBuffers;
namespace mtl {
struct ComputeChain;
} // namespace mtl

struct SyncResult {
    std::vector<state::Entity> NewlyInserted;
    std::vector<state::Entity> NewMeshEntities;
    std::vector<state::Entity> NewExtrasEntities;
    bool SlotsChanged{false}; // An instance slot was inserted, removed or moved.
};

// Moves the mesh's contribution to the flag totals to its record's flags, visible instance count and meshlet work.
void RetallyMesh(state::Registry &, state::Entity mesh_entity);
// Points the meshes' instances at their current render records and refreshes their meshlet work.
// Each mesh's layout and record display fields rederive at the end of the settle pass.
void RepointMeshInstances(state::Registry &, std::span<const state::Entity>);
// Builds and places level-zero meshlets for the meshes and the procedural bone meshes in one batch on the chain, in input order.
// The build submits the chain for its counts and publication, and the refreshed LOD attributes run with its next submit.
void BuildMeshlets(state::Registry &, mtl::ComputeChain &, std::span<const state::Entity> mesh_entities, std::span<const state::Entity> bone_entities);
// Recomputes each material's required LOD attributes and returns whether any differ from the attributes the primitives contain.
bool RefreshMaterialLodAttributes(state::Registry &);
// Refreshes primitive dependencies and records the refit of coarse groups those changes invalidate.
void RefreshClusterLodAttributes(state::Registry &, mtl::ComputeChain &, std::span<const state::Entity>);
// Builds the cluster hierarchy for each face mesh that lacks one and that an unpinned instance draws, and returns whether any mesh took one.
// The primitives' refreshed LOD attributes record on the chain.
bool BuildDemandedClusterLods(state::Registry &, mtl::ComputeChain &);
// Keeps one state buffer and one light buffer per shown scene, fills the states of each newly shown scene, and retallies every mesh when the shown scenes change.
// Returns whether the shown scenes changed.
bool SyncShownScenes(state::Registry &);
// The scene's position among the shown scenes, or zero when the scene is not shown.
uint32_t ShownSceneIndex(const state::Registry &, state::Entity scene);
// Derives a RenderInstance for the Instance of each entity in RenderChanges::InstanceVisibility, and writes its hidden bit into a placed instance's state in each shown scene.
// A rescope rewrites the hidden bit of every placed instance.
// Returns whether a placed instance's visibility changed.
bool DeriveRenderInstances(state::Registry &, bool rescope);
// Places new render instances and compacts removed ones, and marks each buffer entity whose slots changed for layout.
SyncResult SyncModelsBuffers(state::Registry &);
// Sizes the viewport's targets to the render extent once the live consumer finishes sampling them, keeps the transmission target its lighting needs, and binds the sampler slot of each target that changed.
// Returns whether a target changed.
bool SyncViewportRenderResources(state::Registry &, state::Entity viewport, numeric::uvec2 render_extent);
// Whether the node draws hidden in the scene: Hidden, or outside the scene.
bool HiddenIn(const state::Registry &, state::Entity, state::Entity scene);
// The instance's Selected, Active and Hidden state bits.
uint8_t InstanceStateBits(const state::Registry &, state::Entity);
// Writes the base state into the slot of each shown scene's states, with the hidden bit also set in each scene the instance is outside.
void WriteInstanceState(const state::Registry &, GpuBuffers &, state::Entity, uint32_t slot, uint8_t base);
// Whether a mesh's selected instances outline through the screen-space silhouette, which face meshes do and extras, bones, and joints do not.
bool IsSilhouetteEligible(const state::Registry &, state::Entity mesh_entity);
