#pragma once

#include <entt/entity/fwd.hpp>

#include <filesystem>
#include <span>

struct ObjPlyMaterial;

// Upload textures, append materials to the GPU material buffer, register their names,
// and remap the mesh's per-primitive material indices to the appended slots.
void ImportObjPlyMaterials(entt::registry &, std::span<const ObjPlyMaterial>, const std::filesystem::path &mesh_path, uint32_t mesh_store_id);

// Release all imported texture sampler slots and reset to the default white texture and material.
void ResetImportedTexturesAndMaterials(entt::registry &);
