#pragma once
#include "state/Entity.h"
#include <span>
#include <vector>
struct Mesh;
struct GpuBuffers;
struct MeshBuffers;

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
void WriteElementIndices(GpuBuffers &, const Mesh &, MeshBuffers &);
SyncResult SyncModelsBuffers(state::Scene &);
bool SyncViewportRenderResources(state::Scene &, state::Entity);
uint8_t InstanceStateBits(const state::Scene &, state::Entity);
