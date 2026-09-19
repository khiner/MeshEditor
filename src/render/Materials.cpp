#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"
#include "project/store/History.h"
#include "project/store/Records.h"
#include "render/MaterialComponents.h"
#include "state/Scene.h"

MaterialStore::MaterialStore() = default;
MaterialStore::~MaterialStore() = default;

void MaterialStore::AppendNames(std::vector<std::string> names) {
    if (Tracked) Tracked->Write(Names.size(), names.size());
    Names.insert(Names.end(), std::make_move_iterator(names.begin()), std::make_move_iterator(names.end()));
}

void MaterialStore::ResizeNames(size_t size) {
    if (Tracked) Tracked->Write(std::min(size, Names.size()), std::max(size, Names.size()) - std::min(size, Names.size()));
    Names.resize(size);
}

std::optional<uint32_t> DisplayedMaterial(const state::Scene &r, state::Entity mesh_entity) {
    const auto *slot = r.try_get<const MeshMaterialSlotSelection>(mesh_entity);
    if (!slot) return {};
    if (const auto *pending = r.try_get<const MeshMaterialAssignment>(mesh_entity); pending && pending->PrimitiveIndex == slot->PrimitiveIndex) return pending->MaterialIndex;
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto materials = meshes.Arenas().PrimitiveMaterials.Get(meshes.Get(GetMesh(r, mesh_entity).GetStoreId()).PrimitiveMaterials);
    if (slot->PrimitiveIndex >= materials.size()) return {};
    return materials[slot->PrimitiveIndex];
}

void MaterialStore::Track(store::History &history) {
    Tracked = std::make_unique<store::Records>(Names);
    history.Track(*Tracked, "material.names", 0);
}
