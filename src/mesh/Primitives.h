#pragma once
#include "mesh/MeshCreate.h"
#include "mesh/PrimitiveType.h"

namespace primitive {
// The primitive's polygons, flat shaded, with texture coordinate set 0 per face corner where `texcoords` holds.
MeshSource CreateMesh(const PrimitiveParameters &, bool texcoords = false);

struct BoneOctahedronData {
    MeshData Mesh;
    MeshVertexAttributes Attrs;
    std::vector<uint32_t> AdjacencyIndices;
};
BoneOctahedronData BoneOctahedron(float length = 1.0f);

struct BoneSphereData {
    MeshData Mesh;
    std::vector<uint32_t> OutlineIndices;
};
BoneSphereData BoneSphereDisc(float radius = 0.05f, uint32_t segments = 32);
} // namespace primitive
