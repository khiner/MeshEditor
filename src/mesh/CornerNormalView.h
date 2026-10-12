#pragma once

#include "gpu/CornerClassMode.h"
#include "gpu/CustomNormal.h"
#include "gpu/NormalSector.h"
#include "gpu/Vertex.h"
#include "mesh/CornerNormalOffset.h"
#include "mesh/ElementAttributeView.h"

// Borrows current base shading sources, indexed by arena address, and the record's layers, indexed by its handles, and resolves the record's polygon corners.
struct CornerNormalView {
    uint32_t ClassMode{uint32_t(CornerClassMode::UniformVertex)};
    std::span<const uint32_t> CornerVertices;
    std::span<const Vertex> Vertices;
    std::span<const vec3> VertexNormals, FaceNormals;
    MeshConnectivity Connectivity;
    std::span<const uint8_t> FaceSharpness;
    ElementAttributeView<uint32_t> CornerSectors;
    ElementAttributeView<NormalSector> NormalSectors;
    ElementAttributeView<CustomNormal> CustomNormals;

    // The normal of one of the record's corners, whose handles the connectivity resolves to the spans' arena addresses.
    // A packed soup without connectivity indexes its uniform vertex normals by handle.
    vec3 operator[](uint32_t corner) const {
        const bool mixed = ClassMode == uint32_t(CornerClassMode::Mixed);
        const bool uniform_face = ClassMode == uint32_t(CornerClassMode::UniformFace);
        if (Connectivity.HalfedgeTable.empty()) return VertexNormals[CornerVertices[corner]];
        const auto address = Connectivity.HalfedgeAddress(corner).Value();
        const uint32_t face = mixed || uniform_face ? Connectivity.FaceAddress(*Connectivity.FaceOf(Mesh::HH{corner})).Value() : InvalidOffset;
        const bool flat = uniform_face || (mixed && FaceSharpness[face]);
        const auto root = mixed && !flat ? CornerSectors.GetOr(corner, InvalidOffset) : InvalidOffset;
        vec3 normal = flat ? FaceNormals[face] : root == InvalidOffset ? VertexNormals[Connectivity.VertexAddress(CornerVertices[address]).Value()] :
                                                                         NormalSectors[root].Normal;
        const auto offset = CustomNormals.GetOr(corner).Offset;
        if (offset.x >= 0.f) {
            const Mesh::HH h{corner};
            const auto position = [&](Mesh::HH at) { return Vertices[Connectivity.VertexAddress(CornerVertices[Connectivity.HalfedgeAddress(*at).Value()]).Value()].Position; };
            normal = DecodeNormalOffset(offset, ComputeCornerFrame(normal, position(h), position(Connectivity.Next(h)), position(Connectivity.Previous(h))));
        }
        return normal;
    }
};
