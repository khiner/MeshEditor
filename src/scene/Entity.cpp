#include "scene/Entity.h"
#include "state/Scene.h"

#include "mesh/Mesh.h"
#include "render/Instance.h"

#include <charconv>
#include <format>
#include <unordered_map>

namespace {
struct NameHash {
    using is_transparent = void;
    size_t operator()(std::string_view name) const { return std::hash<std::string_view>{}(name); }
};
template<typename V> using NameMap = std::unordered_map<std::string, V, NameHash, std::equal_to<>>;

// Counts rather than set membership keep the derived index correct when loading source data with duplicate names.
struct EntityNameCounts {
    NameMap<size_t> Counts;
    // Per stem, at least the largest N of every live "{stem}_{N}" name.
    NameMap<uint64_t> MaxSuffix;
};

void Track(EntityNameCounts &names, std::string_view name) {
    if (const auto it = names.Counts.find(name); it != names.Counts.end()) ++it->second;
    else names.Counts.emplace(name, 1u);
    const auto stem = NameStem(name);
    if (stem.size() == name.size()) return;
    uint64_t suffix = 0;
    if (std::from_chars(name.data() + stem.size() + 1u, name.data() + name.size(), suffix).ec != std::errc{}) return;
    if (const auto it = names.MaxSuffix.find(stem); it != names.MaxSuffix.end()) it->second = std::max(it->second, suffix);
    else names.MaxSuffix.emplace(stem, suffix);
}

void TrackName(state::Scene &r, state::Entity e) {
    if (auto *names = r.Context.find<EntityNameCounts>()) Track(*names, r.get<const Name>(e).Value);
}
void UntrackName(state::Scene &r, state::Entity e) {
    auto *names = r.Context.find<EntityNameCounts>();
    if (!names) return;
    const auto it = names->Counts.find(r.get<const Name>(e).Value);
    if (it != names->Counts.end() && --it->second == 0) names->Counts.erase(it);
}
} // namespace

void InitEntityNames(state::Scene &r) {
    r.Context.emplace<EntityNameCounts>();
    r.on_construct<Name, &TrackName>();
    r.on_destroy<Name, &UntrackName>();
}
void RebuildEntityNames(state::Scene &r) {
    auto &names = r.Context.get<EntityNameCounts>();
    names.Counts.clear();
    names.MaxSuffix.clear();
    for (const auto &[e, name] : r.view<const Name>().each()) Track(names, name.Value);
}
void DeinitEntityNames(state::Scene &r) { r.Context.erase<EntityNameCounts>(); }
void ReserveEntityNames(state::Scene &r, size_t additional) {
    auto &counts = r.Context.get<EntityNameCounts>().Counts;
    counts.reserve(counts.size() + additional);
}
Name &EmplaceUniqueName(state::Scene &r, state::Entity e, std::string_view prefix) {
    const auto &names = r.Context.get<const EntityNameCounts>();
    if (!names.Counts.contains(prefix)) return r.emplace<Name>(e, std::string{prefix});
    const auto it = names.MaxSuffix.find(prefix);
    for (auto suffix = (it != names.MaxSuffix.end() ? it->second : 0u) + 1u;; ++suffix)
        if (auto candidate = std::format("{}_{}", prefix, suffix); !names.Counts.contains(candidate)) return r.emplace<Name>(e, std::move(candidate));
}

std::string_view NameStem(std::string_view name) {
    const auto underscore = name.find_last_of('_');
    if (underscore == std::string_view::npos || underscore == 0u || underscore + 1u == name.size()) return name;
    const bool numbered = std::ranges::all_of(name.substr(underscore + 1u), [](char c) { return c >= '0' && c <= '9'; });
    return numbered ? name.substr(0u, underscore) : name;
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
