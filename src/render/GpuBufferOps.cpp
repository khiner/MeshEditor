#include "render/GpuBufferOps.h"

#include "render/GpuBuffers.h"
#include "render/MeshBuffers.h"

#include <entt/entity/registry.hpp>

std::span<PBRMaterial> GetMaterials(entt::registry &r) {
    auto &materials = r.ctx().get<GpuBuffers>().Materials;
    return {materials.Data(), materials.Count()};
}
std::span<const PunctualLight> GetLights(entt::registry &r) {
    const auto &lights = r.ctx().get<GpuBuffers>().Lights;
    return {lights.Data(), lights.Count()};
}
WorkspaceLights &GetWorkspaceLights(entt::registry &r) { return r.ctx().get<GpuBuffers>().GetWorkspaceLights(); }
PunctualLight GetLight(entt::registry &r, uint32_t index) { return r.ctx().get<GpuBuffers>().Lights.Get(index); }
mvk::BufferContext &GetBufferContext(entt::registry &r) { return r.ctx().get<GpuBuffers>().Ctx; }

void ReleaseMeshBuffers(entt::registry &r, MeshBuffers &mb) { r.ctx().get<GpuBuffers>().Release(mb); }

uint32_t AllocateVertexClasses(entt::registry &r, std::span<const uint8_t> classes) {
    return r.ctx().get<GpuBuffers>().VertexClassBuffer.Allocate(classes).Offset;
}
void ReleaseVertexClasses(entt::registry &r, uint32_t offset, uint32_t count) {
    r.ctx().get<GpuBuffers>().VertexClassBuffer.Release({offset, count});
}
void FreeInstanceRange(entt::registry &r, Range range) { r.ctx().get<GpuBuffers>().Instances.Free(range); }
void ReleaseEdgeIndices(entt::registry &r, const SlottedRange &indices) { r.ctx().get<GpuBuffers>().EdgeIndexBuffer.Release(indices); }
