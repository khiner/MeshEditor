#pragma once
#include "mesh/MeshAttributes.h"
#include "mesh/MeshData.h"
#include "mesh/PrimitiveType.h"

namespace primitive {
MeshData CreateMesh(const Plane &);
MeshData CreateMesh(const Circle &);
MeshData CreateMesh(const Cuboid &);
MeshData CreateMesh(const IcoSphere &);
MeshData CreateMesh(const UVSphere &);
MeshData CreateMesh(const Torus &);
MeshData CreateMesh(const Cylinder &);
MeshData CreateMesh(const Cone &);
MeshData CreateMesh(const PrimitiveShape &);

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
