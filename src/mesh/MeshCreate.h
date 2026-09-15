#pragma once

#include "mesh/MeshAttributes.h"
#include "mesh/MeshData.h"
#include "mesh/MeshStore.h"

#include <span>
#include <vector>

#include "state/Entity.h"

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

// MorphTangentDeltas returns the target-major tangent deltas the arena doesn't store, compacted to the welded vertex set.
// AuthoredCornerNormals returns a triangle mesh's authored normals in fan order for EncodeAuthoredCornerNormals once its base normals derive.
struct CreatedMesh {
    uint32_t StoreId;
    std::vector<vec3> MorphTangentDeltas{};
    std::vector<vec3> AuthoredCornerNormals{};
};

// The arena work of each phase runs in source order, so a batch lays out the same way every run.
// Authored normals recover as face sharpness on faceted faces, as edge sharpness where they disagree across an edge, and otherwise as a custom corner-normal layer.
std::vector<CreatedMesh> CreateMeshes(state::Scene &, std::span<MeshSource>);
CreatedMesh CreateMesh(state::Scene &, MeshSource);

// Encode the authored corner normals as offsets from the derived corner normals where they deviate, filling the custom corner-normal layer.
// Requires derived base normals and an empty custom layer.
void EncodeAuthoredCornerNormals(MeshStore &, const Mesh &, std::span<const vec3> authored);

// Preserves authored shading when targets include normal deltas or any listed full-weight pose materially changes the derived corner normals.
// Requires derived base normals.
void UpdateMorphShadingAuthored(MeshStore &, const Mesh &, std::span<const CornerNormalSources> poses);
