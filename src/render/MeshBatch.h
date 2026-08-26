#pragma once

#include "mesh/MeshAttributes.h"
#include "mesh/MeshData.h"
#include "mesh/MeshStore.h"

#include <span>
#include <vector>

#include <entt/entity/fwd.hpp>

// `Weld` merges vertices identical in every vertex-domain channel, which a source authored per corner needs.
struct MeshSource {
    MeshData Data;
    MeshVertexAttributes Attrs{};
    MeshPrimitives Primitives{};
    std::optional<ArmatureDeformData> Deform{};
    std::optional<MorphTargetData> Morph{};
    bool Weld{false};
    bool FlatShaded{false};
};

// Every mesh the store makes comes through here.
// The arena work of each phase runs in source order, so a batch lays out the same way every run.
std::vector<CreatedMesh> CreateMeshes(entt::registry &, std::span<MeshSource>);
CreatedMesh CreateMesh(entt::registry &, MeshSource);
