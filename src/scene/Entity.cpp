#include "scene/Entity.h"
#include "state/Scene.h"

#include "mesh/Mesh.h"
#include "render/Instance.h"

#include <format>
#include <limits>
#include <unordered_map>

namespace {
// Counts rather than set membership keep the derived index correct when loading source data with duplicate names.
struct EntityNameCounts {
    std::unordered_map<std::string, size_t> Counts;
};

void TrackName(state::Scene &r, state::Entity e) {
    if (auto *names = r.Context.find<EntityNameCounts>()) ++names->Counts[r.get<const Name>(e).Value];
}
void UntrackName(state::Scene &r, state::Entity e) {
    auto *names = r.Context.find<EntityNameCounts>();
    if (!names) return;
    const auto it = names->Counts.find(r.get<const Name>(e).Value);
    if (it != names->Counts.end() && --it->second == 0) names->Counts.erase(it);
}

std::string ChooseUniqueName(const state::Scene &r, std::string_view prefix) {
    const auto &counts = r.Context.get<const EntityNameCounts>().Counts;
    const std::string base{prefix};
    for (uint32_t i = 0; i < std::numeric_limits<uint32_t>::max(); ++i) {
        auto candidate = i == 0 ? base : std::format("{}_{}", prefix, i);
        if (!counts.contains(candidate)) return candidate;
    }
    assert(false);
    return base;
}
} // namespace

void InitEntityNames(state::Scene &r) {
    r.Context.emplace<EntityNameCounts>();
    r.on_construct<Name, &TrackName>();
    r.on_destroy<Name, &UntrackName>();
}
void RebuildEntityNames(state::Scene &r) {
    auto &counts = r.Context.get<EntityNameCounts>().Counts;
    counts.clear();
    for (const auto &[e, name] : r.view<const Name>().each()) ++counts[name.Value];
}
void DeinitEntityNames(state::Scene &r) { r.Context.erase<EntityNameCounts>(); }
void ReserveEntityNames(state::Scene &r, size_t additional) {
    auto &counts = r.Context.get<EntityNameCounts>().Counts;
    counts.reserve(counts.size() + additional);
}
Name &EmplaceUniqueName(state::Scene &r, state::Entity e, std::string_view prefix) {
    return r.emplace<Name>(e, ChooseUniqueName(r, prefix));
}

std::string IdString(state::Entity e) { return std::format("0x{:08x}", uint32_t(e)); }
std::string GetName(const state::Scene &r, state::Entity e) {
    if (e == state::Null) return "null";

    if (const auto *name = r.try_get<Name>(e)) {
        if (!name->Value.empty()) return name->Value;
    }
    return IdString(e);
}

state::Entity FindActiveEntity(const state::Scene &registry) {
    auto all_active = registry.view<Active>();
    assert(all_active.size() <= 1);
    return all_active.empty() ? state::Null : *all_active.begin();
}

state::Entity GetMeshEntity(const state::Scene &r, state::Entity e) {
    if (const auto *instance = r.try_get<Instance>(e); instance && HasMesh(r, instance->Entity)) return instance->Entity;
    return state::Null;
}
state::Entity GetActiveMeshEntity(const state::Scene &r) {
    const auto active = FindActiveEntity(r);
    return active != state::Null ? GetMeshEntity(r, active) : state::Null;
}

state::Entity FindMeshEntity(const state::Scene &r, state::Entity entity) {
    if (const auto *instance = r.try_get<const Instance>(entity)) return instance->Entity;
    return entity;
}
