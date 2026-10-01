#pragma once

#include "state/Entity.h"

#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

struct ObjPlyMaterial;

// Read and decode referenced textures before uploading textures and materials.
// Return the appended GPU material slots in source material order.
std::expected<std::vector<uint32_t>, std::string> ImportObjPlyMaterials(state::Scene &, state::Entity viewport, std::span<const ObjPlyMaterial>, const std::filesystem::path &mesh_path);
