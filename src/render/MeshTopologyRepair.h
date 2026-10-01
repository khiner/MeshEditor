#pragma once
#include "state/Entity.h"
#include <cstdint>
#include <span>
struct MeshTopologyEdit;
struct FaceTriangles;
struct MeshBuffers;
struct MeshletBuildSource;
namespace mtl { struct ComputeChain; }

// Publish a canonical topology edit through finest payload, simplification
// dependencies and traversal bounds before retiring its source identities.
// The affected coarse groups turn stale until the next coarse repair.
// The repair records into the edit's chain, which submits where the host reads counts and records, and whose owner submits the rest.
// Fresh builds of other meshes ride the repair's meshlet build, which publishes their element owners.
void RepairTopologyRender(state::Scene &,state::Entity,const MeshTopologyEdit &,std::span<MeshletBuildSource> fresh = {});
void RepairShadingRender(state::Scene &,mtl::ComputeChain &,state::Entity,const FaceTriangles &);
// Moves the affected vertices of a point record, or the affected edges of a line record, between its owner's clusters, which draw every live element.
// Retired elements leave their clusters, and live elements without one build as fragments the owner adopts, recorded on the chain.
void RepairElementMeshlets(state::Scene &, mtl::ComputeChain &, MeshBuffers &owner, std::span<const uint32_t> elements);
