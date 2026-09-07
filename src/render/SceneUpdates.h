#pragma once
#include "entt_fwd.h"
#include <span>
#include <vector>
struct Mesh;
struct GpuBuffers;
struct MeshBuffers;

struct SyncResult {
    std::vector<entt::entity> NewlyInserted;
    std::vector<entt::entity> NewMeshEntities;
    std::vector<entt::entity> NewExtrasEntities;
    bool Compacted{false};
};

void RepointMeshInstances(entt::registry &, std::span<const entt::entity>);
void BuildMeshletsNow(entt::registry &, std::span<const entt::entity>);
void BuildBoneMeshletsNow(entt::registry &, std::span<const entt::entity>);
bool DrawsElementIndices(const entt::registry &, entt::entity);
bool DrawsStoredCorners(const Mesh &);
bool NeedsElementIndices(const Mesh &, bool);
void WriteElementIndices(GpuBuffers &, const Mesh &, MeshBuffers &);
SyncResult SyncModelsBuffers(entt::registry &);
bool SyncViewportRenderResources(entt::registry &, entt::entity);
uint8_t InstanceStateBits(const entt::registry &, entt::entity);
