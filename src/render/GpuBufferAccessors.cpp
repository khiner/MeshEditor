#include "render/GpuBufferAccessors.h"

#include "render/GpuBuffers.h"

#include <entt/entity/registry.hpp>

std::span<PBRMaterial> GetMaterials(entt::registry &r) {
    auto &materials = r.ctx().get<GpuBuffers>().Materials;
    return {materials.Data(), materials.Count()};
}

std::span<const PunctualLight> GetLights(entt::registry &r) {
    const auto &lights = r.ctx().get<GpuBuffers>().Lights;
    return {lights.Data(), lights.Count()};
}

WorkspaceLights &GetWorkspaceLights(entt::registry &r) {
    return r.ctx().get<GpuBuffers>().GetWorkspaceLights();
}
