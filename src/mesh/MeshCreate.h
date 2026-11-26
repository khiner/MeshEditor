#pragma once

#include "mesh/CornerClassBatch.h"
#include "mesh/MeshStore.h"

// `Weld` merges vertices identical in every vertex-domain channel, which a source authored per corner needs.
// A source holds texture coordinate set 0 in `CornerTexCoords0` or in `Attrs.TexCoords0`, and wire corners take the absent value.
struct MeshSource {
    MeshData Data;
    MeshVertexAttributes Attrs{};
    std::vector<vec2> CornerTexCoords0{}; // Per face corner in Data's loop order
    MeshPrimitives Primitives{};
    std::optional<ArmatureDeformData> Deform{};
    std::optional<MorphTargetData> Morph{};
    bool Weld{false};
    bool KeepLooseVertices{true}; // With Weld, keeps vertices no face or wire references
    bool FlatShaded{false};
};

// MorphTangentDeltas holds the target-major tangent deltas the arena does not store, compacted to the welded vertices.
// AuthoredCornerNormals holds authored normals in polygon corner order.
struct CreatedMesh {
    uint32_t StoreId;
    std::vector<vec3> MorphTangentDeltas{};
    std::vector<vec3> AuthoredCornerNormals{};
};

// Creates the records in source order and recovers authored normals as face sharpness, edge sharpness or custom corner normals.
std::vector<CreatedMesh> CreateMeshes(state::Registry &, std::span<MeshSource>);
CreatedMesh CreateMesh(state::Registry &, MeshSource);

// The shading of the records CreateMesh completed since the last derive, between its two recordings.
struct NewMeshShading {
    std::vector<uint32_t> Ids;
    CornerClassBatch Classes;
};
// Takes the records awaiting shading and records their corner classification counts, when any await it.
std::optional<NewMeshShading> EncodeNewMeshShading(state::Registry &, mtl::ComputeChain &);
// Once the chain submitted the counts, records the records' classification writes, base normal derive and selection state.
// The selection summaries publish with the chain's next submit.
void DeriveNewMeshShading(state::Registry &, mtl::ComputeChain &, NewMeshShading &);
// Derives the shading of the records awaiting it on a chain of its own, which submits for the classification plan and once at the end.
void ShadeNewMeshes(state::Registry &);
uint32_t CreateShadedMesh(state::Registry &, MeshSource);

// Writes each authored corner normal that deviates from the derived one as an offset in the custom corner-normal layer.
// Requires derived base normals and an empty custom layer.
void EncodeAuthoredCornerNormals(MeshStore &, const Mesh &, std::span<const vec3> authored);

// Preserves authored shading when targets include normal deltas or any listed full-weight pose materially changes the derived corner normals.
// Requires derived base normals.
void UpdateMorphShadingAuthored(MeshStore &, const Mesh &, std::span<const CornerNormalSources> poses);
