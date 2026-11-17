#include "mesh/MeshStore.h"
#include "render/GpuBuffers.h"
#include "render/MaterialComponents.h"
#include "state/Registry.h"

std::vector<state::Entity> MaterialsByIndex(const state::Registry &r) {
    std::vector<state::Entity> entities(r.Resources.get<const GpuBuffers>().Materials.Count<PBRMaterial>(), state::Null);
    for (const auto [e, material] : r.view<const Material>().each()) entities[material.Index] = e;
    return entities;
}
state::Entity MaterialAt(const state::Registry &r, uint32_t index) {
    for (const auto [e, material] : r.view<const Material>().each()) {
        if (material.Index == index) return e;
    }
    return state::Null;
}

uint32_t MaterialIndexOf(const state::Registry &r, state::Entity material) {
    const auto *record = r.valid(material) ? r.try_get<const Material>(material) : nullptr;
    return record ? record->Index : 0u;
}

state::Entity AddMaterial(state::Registry &r, const PBRMaterial &value, std::string name) {
    auto &materials = r.Resources.get<GpuBuffers>().Materials;
    const auto used = MaterialsByIndex(r);
    // The default record at index 0 is never free.
    const auto index = uint32_t(std::find(used.begin() + 1, used.end(), state::Null) - used.begin());
    if (index == used.size()) materials.Append(value);
    else materials.Update(std::as_bytes(std::span{&value, 1}), uint64_t(index) * sizeof(PBRMaterial));
    const auto e = r.create();
    r.emplace<Material>(e, index, std::move(name));
    return e;
}

std::optional<state::Entity> DisplayedMaterial(const state::Registry &r, state::Entity mesh_entity) {
    const auto *slot = r.try_get<const MeshMaterialSlotSelection>(mesh_entity);
    if (!slot) return {};
    if (const auto *pending = r.try_get<const MeshMaterialAssignment>(mesh_entity); pending && pending->PrimitiveIndex == slot->PrimitiveIndex) return pending->Material;
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto materials = meshes.Arenas().PrimitiveMaterials.Get(meshes.Get(GetMesh(r, mesh_entity).GetStoreId()).PrimitiveMaterials);
    if (slot->PrimitiveIndex >= materials.size()) return {};
    return MaterialAt(r, materials[slot->PrimitiveIndex]);
}
