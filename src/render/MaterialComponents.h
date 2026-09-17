#pragma once

#include "state/Entity.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace state {
struct Scene;
} // namespace state

namespace store {
struct History;
struct Records;
} // namespace store

struct MaterialStore {
    MaterialStore();
    ~MaterialStore();
    void AppendNames(std::vector<std::string>);
    void ResizeNames(size_t);
    void Track(store::History &);
    std::vector<std::string> Names;

private:
    std::unique_ptr<store::Records> Tracked;
};

struct MeshMaterialAssignment {
    uint32_t PrimitiveIndex, MaterialIndex;
};
struct MeshMaterialSlotSelection {
    uint32_t PrimitiveIndex{0};
};

// The material the mesh entity's selected slot shows: its pending assignment, else the mesh's primitive material.
std::optional<uint32_t> DisplayedMaterial(const state::Scene &, state::Entity mesh_entity);

// Presence indicates declared glTF variants.
// An empty Active value selects each primitive's default material.
struct MaterialVariants {
    std::vector<std::string> Names;
    std::optional<uint32_t> Active;
};

// Per-mesh-entity: bitmask of PbrFeature bits that are explicitly enabled for that mesh.
// Scene-wide mask = OR of all PbrMeshFeatures + Punctual bit from "Use Scene Lights".
struct PbrMeshFeatures {
    uint32_t Mask{0};
};
