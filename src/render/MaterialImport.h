#pragma once

#include <entt/entity/fwd.hpp>

#include <filesystem>
#include <span>

struct ObjPlyMaterial;

// Uploads textures and materials and remaps primitive material indices to the appended GPU slots.
void ImportObjPlyMaterials(entt::registry &, entt::entity viewport, std::span<const ObjPlyMaterial>, const std::filesystem::path &mesh_path, uint32_t mesh_store_id);

// Release imported GPU textures while retaining the default white texture.
void ReleaseImportedTextures(entt::registry &);
// Release imported textures and reset to the default material.
void ResetImportedTexturesAndMaterials(entt::registry &);
