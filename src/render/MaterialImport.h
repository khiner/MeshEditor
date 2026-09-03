#pragma once

#include <entt/entity/fwd.hpp>

#include <filesystem>
#include <span>

struct ObjPlyMaterial;

// Uploads textures and materials and remaps primitive material indices to the appended GPU slots.
void ImportObjPlyMaterials(entt::registry &, std::span<const ObjPlyMaterial>, const std::filesystem::path &mesh_path, uint32_t mesh_store_id);

// Release all imported texture sampler slots and reset to the default white texture and material.
void ResetImportedTexturesAndMaterials(entt::registry &);
