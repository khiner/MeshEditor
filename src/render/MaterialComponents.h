#pragma once

#include "render/RenderSchema.h"
#include "state/Entity.h"

#include <string>
#include <vector>

struct PBRMaterial;

// A material node: the index of its record in the material buffer, and its name.
// Index 0 is the default record, which draws a primitive without a material and has no node.
struct Material {
    uint32_t Index;
    std::string Name;
};
// Each material's entity by record index, with Null at the default record and at free indices.
std::vector<state::Entity> MaterialsByIndex(const state::Registry &);
state::Entity MaterialAt(const state::Registry &, uint32_t index);
// The material's record index, or the default record for no material or a deleted one.
uint32_t MaterialIndexOf(const state::Registry &, state::Entity material);
// Creates a material whose record takes the lowest free index.
state::Entity AddMaterial(state::Registry &, const PBRMaterial &, std::string name);

// A material assignment the next event pass writes to the mesh's primitive materials, where no material is the default record.
struct MeshMaterialAssignment {
    uint32_t PrimitiveIndex;
    state::Entity Material;
};
struct MeshMaterialSlotSelection {
    uint32_t PrimitiveIndex{0};
};

// The material the mesh entity's selected slot shows: its pending assignment, else the mesh's primitive material, where null is no material.
std::optional<state::Entity> DisplayedMaterial(const state::Registry &, state::Entity mesh_entity);

// Presence indicates declared glTF variants.
// An empty Active value selects each primitive's default material.
struct MaterialVariants {
    std::vector<std::string> Names;
    std::optional<uint32_t> Active;
};

// Per-mesh-entity: bitmask of PbrFeature bits that are explicitly enabled for that mesh.
// Scene-wide mask = OR of all PbrMeshFeatures + SceneLights bit from "Use Scene Lights".
struct PbrMeshFeatures {
    uint32_t Mask{0};
};
