#pragma once

#include "mesh/CornerClassBatch.h"
#include "mesh/MeshStore.h"
#include "mesh/NormalDeriveGpu.h"

// `Weld` merges vertices identical in every vertex-domain channel, which a source authored per corner needs.
struct MeshSource {
    MeshData Data;
    MeshVertexAttributes Attrs{};
    MeshPrimitives Primitives{};
    std::optional<ArmatureDeformData> Deform{};
    std::optional<MorphTargetData> Morph{};
    bool Weld{false};
    bool KeepLooseVertices{true}; // When welding, retain vertices not referenced by faces or wires.
    bool FlatShaded{false};
};

// MorphTangentDeltas returns the target-major tangent deltas the arena doesn't store, compacted to the welded vertex set.
// AuthoredCornerNormals returns authored normals in polygon-corner order for EncodeAuthoredCornerNormals once its base normals derive.
struct CreatedMesh {
    uint32_t StoreId;
    std::vector<vec3> MorphTangentDeltas{};
    std::vector<vec3> AuthoredCornerNormals{};
};

// Process arena phases in source order for deterministic layout.
// Recover authored normals as face sharpness, edge sharpness, or custom corner normals.
// Derive classification, base normals, and selection in the later new-mesh shading pass.
std::vector<CreatedMesh> CreateMeshes(state::Scene &, std::span<MeshSource>);
CreatedMesh CreateMesh(state::Scene &, MeshSource);

// The shading of the records CreateMesh completed since the last derive, between its two recordings.
struct NewMeshShading {
    std::vector<uint32_t> Ids;
    CornerClassBatch Classes;
    AllNormalsWork Normals;
};
// Takes the records awaiting shading and records their corner classification counts and their normal membership gather, when any await it.
std::optional<NewMeshShading> EncodeNewMeshShading(state::Scene &, mtl::ComputeChain &);
// Once the chain submitted the counts, records the records' classification writes, base normal derive and selection state.
// The selection summaries publish with the chain's next submit.
void DeriveNewMeshShading(state::Scene &, mtl::ComputeChain &, NewMeshShading &);
// Derives the shading of the records awaiting it on a chain of its own, which submits for the classification plan and once at the end.
void ShadeNewMeshes(state::Scene &);

// Encode the authored corner normals as offsets from the derived corner normals where they deviate, filling the custom corner-normal layer.
// Requires derived base normals and an empty custom layer.
void EncodeAuthoredCornerNormals(MeshStore &, const Mesh &, std::span<const vec3> authored);

// Preserves authored shading when targets include normal deltas or any listed full-weight pose materially changes the derived corner normals.
// Requires derived base normals.
void UpdateMorphShadingAuthored(MeshStore &, const Mesh &, std::span<const CornerNormalSources> poses);
