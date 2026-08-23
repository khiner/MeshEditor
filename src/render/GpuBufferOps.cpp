#include "render/GpuBufferOps.h"

#include "render/GpuBuffers.h"
#include "render/MeshBuffers.h"

#include <entt/entity/registry.hpp>

std::span<PBRMaterial> GetMaterials(entt::registry &r) {
    auto &materials = r.ctx().get<GpuBuffers>().Materials;
    return {materials.Data(), materials.Count()};
}
std::span<const uint32_t> GetFaceIndices(const entt::registry &r, const MeshBuffers &buffers) {
    return r.ctx().get<const GpuBuffers>().FaceIndexBuffer.Get(buffers.FaceIndices);
}
std::span<const PunctualLight> GetLights(entt::registry &r) {
    const auto &lights = r.ctx().get<GpuBuffers>().Lights;
    return {lights.Data(), lights.Count()};
}
PunctualLight GetLight(entt::registry &r, uint32_t index) { return r.ctx().get<GpuBuffers>().Lights.Get(index); }
mtl::BufferContext &GetBufferContext(entt::registry &r) { return r.ctx().get<GpuBuffers>().Ctx; }

void ReleaseMeshBuffers(entt::registry &r, MeshBuffers &mb) { r.ctx().get<GpuBuffers>().Release(mb); }

void FreeInstanceRange(entt::registry &r, Range range) { r.ctx().get<GpuBuffers>().Instances.Free(range); }
void ReleaseEdgeIndices(entt::registry &r, const SlottedRange &indices) { r.ctx().get<GpuBuffers>().EdgeIndexBuffer.Release(indices); }
