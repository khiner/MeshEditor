#include "Entity.h"

#include "Instance.h"
#include "mesh/Mesh.h"

#include <entt/entity/registry.hpp>

#include <format>

std::string IdString(entt::entity e) { return std::format("0x{:08x}", uint32_t(e)); }
std::string GetName(const entt::registry &r, entt::entity e) {
    if (e == entt::null) return "null";

    if (const auto *name = r.try_get<Name>(e)) {
        if (!name->Value.empty()) return name->Value;
    }
    return IdString(e);
}

entt::entity FindActiveEntity(const entt::registry &registry) {
    auto all_active = registry.view<Active>();
    assert(all_active.size() <= 1);
    return all_active.empty() ? entt::null : *all_active.begin();
}

entt::entity GetMeshEntity(const entt::registry &r, entt::entity e) {
    if (const auto *instance = r.try_get<Instance>(e); instance && r.all_of<Mesh>(instance->Entity)) return instance->Entity;
    return entt::null;
}
entt::entity GetActiveMeshEntity(const entt::registry &r) {
    const auto active = FindActiveEntity(r);
    return active != entt::null ? GetMeshEntity(r, active) : entt::null;
}

entt::entity FindMeshEntity(const entt::registry &r, entt::entity entity) {
    if (const auto *instance = r.try_get<const Instance>(entity)) return instance->Entity;
    return entity;
}
