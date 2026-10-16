#ifndef CORNERRENDERKEY_MSL
#define CORNERRENDERKEY_MSL
#include "CornerNormalOffset.metal"
#include "gpu/CornerClass.h"
#include "gpu/CornerClassMode.h"

// Shared render-vertex equivalence for finest construction and local LOD simplification.
// Callers supply corner handles of the mesh, including corners retained from different source faces in a coarse triangle.
struct CornerRenderKey {
    device const BindlessSet &B;
    MeshRecord Mesh;
    uint FaceSharpnessSlot;
    uint Address(uint corner) const { return HandleAddress<MeshElementBlockSize>(B, Mesh.Connectivity.Tables.Halfedges, corner); }
    uint VertexIndex(uint corner) const {
        return BindlessBuffer(uint,B.IndexBuffer,Mesh.IndexSlotOffset.Slot)[Address(corner)];
    }
    uint Face(uint corner) const {
        return BindlessBuffer(uint,B.Buffer,Mesh.Connectivity.HalfedgeFacesSlot)[Address(corner)];
    }
    uint Class(uint c) const {
        if (Mesh.CornerClassMode == uint(CornerClassMode::UniformVertex)) return uint(CornerClass::Vertex);
        if (Mesh.CornerClassMode == uint(CornerClassMode::UniformFace)) return uint(CornerClass::Face);
        if (BindlessBuffer(uchar, B.Buffer, FaceSharpnessSlot)[HandleAddress<MeshElementBlockSize>(B, Mesh.Connectivity.Tables.Faces, Face(c))] != 0u) return uint(CornerClass::Face);
        return uint(CornerSectorRoot(B, Mesh.Layers.CornerSectors, c) == InvalidOffset ? CornerClass::Vertex : CornerClass::Seam);
    }
    uint ClassIdentity(uint c, uint kind, bool flat) const {
        if (kind == uint(CornerClass::Seam)) return CornerSectorRoot(B, Mesh.Layers.CornerSectors, c);
        return !flat && kind == uint(CornerClass::Face) ? Face(c) : InvalidOffset;
    }
    bool Custom(uint c) const {
        return CustomNormalOffset(B, Mesh.Layers.CustomNormals, c).x >= 0.f;
    }
    bool Flat(uint3 corners) const {
        if (Mesh.MorphShadingAuthored != 0u) return false;
        for (uint c = 0u; c < 3u; ++c) {
            const uint corner = corners[c];
            if (Class(corner) != uint(CornerClass::Face) || Custom(corner)) return false;
        }
        return true;
    }
    uint AttributeWord(uint c, uint word) const {
        const ElementLayers layers = Mesh.Layers;
        uint at = 0u;
        for (uint set = 0u; set < 4u; ++set) {
            if (layers.CornerUvs[set].Slot == InvalidSlot) continue;
            if (word < at + 2u) return as_type<uint>(BindlessBuffer(packed_float2, B.CornerUvBuffer, layers.CornerUvs[set].Slot)[layers.CornerUvs[set].Offset + c][word - at]);
            at += 2u;
        }
        if (layers.CornerTangent.Slot != InvalidSlot) {
            if (word < at + 4u) return as_type<uint>(BindlessBuffer(packed_float4, B.CornerTangentBuffer, layers.CornerTangent.Slot)[layers.CornerTangent.Offset + c][word - at]);
            at += 4u;
        }
        return as_type<uint>(BindlessBuffer(packed_float4, B.CornerColorBuffer, layers.CornerColor.Slot)[layers.CornerColor.Offset + c][word - at]);
    }
    uint AttributeWords() const {
        const ElementLayers layers = Mesh.Layers;
        uint n = (layers.CornerTangent.Slot == InvalidSlot ? 0u : 4u) + (layers.CornerColor.Slot == InvalidSlot ? 0u : 4u);
        for (uint set = 0u; set < 4u; ++set) if (layers.CornerUvs[set].Slot != InvalidSlot) n += 2u;
        return n;
    }
    uint Hash(uint c, bool flat) const {
        const uint value = Class(c);
        uint h = ((VertexIndex(c) * 0x9e3779b1u) ^ value) * 16777619u ^ ClassIdentity(c, value, flat);
        if (Custom(c)) h ^= c * 0x85ebca6bu;
        for (uint w = 0u; w < AttributeWords(); ++w) h = (h ^ AttributeWord(c, w)) * 16777619u;
        h ^= h >> 16u; h *= 0x7feb352du; h ^= h >> 15u;
        return h;
    }
    bool Equal(uint a, bool af, uint b, bool bf) const {
        const uint ac = Class(a), bc = Class(b);
        if (ac != bc || ClassIdentity(a, ac, af) != ClassIdentity(b, bc, bf)) return false;
        if (a == b) return true;
        if (Custom(a) || Custom(b)) return false;
        if (VertexIndex(a) != VertexIndex(b)) return false;
        for (uint w = 0u; w < AttributeWords(); ++w) if (AttributeWord(a, w) != AttributeWord(b, w)) return false;
        return true;
    }
};
#endif
