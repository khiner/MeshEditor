#include "scene/Entity.h"

#include "mesh/Mesh.h"
#include "render/Instance.h"

#include <entt/entity/registry.hpp>

#include <format>
#include <limits>
#include <unordered_map>

namespace {
// Counts rather than set membership keep the derived index correct when loading source or legacy data with duplicate names.
struct EntityNameCounts {
    std::unordered_map<std::string, size_t> Counts;
};

void TrackName(entt::registry &r, entt::entity e) {
    if (auto *names = r.ctx().find<EntityNameCounts>()) ++names->Counts[r.get<const Name>(e).Value];
}
void UntrackName(entt::registry &r, entt::entity e) {
    auto *names = r.ctx().find<EntityNameCounts>();
    if (!names) return;
    const auto it = names->Counts.find(r.get<const Name>(e).Value);
    if (it != names->Counts.end() && --it->second == 0) names->Counts.erase(it);
}

std::string ChooseUniqueName(const entt::registry &r, std::string_view prefix) {
    const auto &counts = r.ctx().get<const EntityNameCounts>().Counts;
    const std::string base{prefix};
    for (uint32_t i = 0; i < std::numeric_limits<uint32_t>::max(); ++i) {
        auto candidate = i == 0 ? base : std::format("{}_{}", prefix, i);
        if (!counts.contains(candidate)) return candidate;
    }
    assert(false);
    return base;
}
} // namespace

void InitEntityNames(entt::registry &r) {
    r.ctx().emplace<EntityNameCounts>();
    r.on_construct<Name>().connect<&TrackName>();
    r.on_destroy<Name>().connect<&UntrackName>();
}
void DeinitEntityNames(entt::registry &r) { r.ctx().erase<EntityNameCounts>(); }
void ReserveEntityNames(entt::registry &r, size_t additional) {
    auto &counts = r.ctx().get<EntityNameCounts>().Counts;
    counts.reserve(counts.size() + additional);
}
Name &EmplaceUniqueName(entt::registry &r, entt::entity e, std::string_view prefix) {
    return r.emplace<Name>(e, ChooseUniqueName(r, prefix));
}

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
    if (const auto *instance = r.try_get<Instance>(e); instance && HasMesh(r, instance->Entity)) return instance->Entity;
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
