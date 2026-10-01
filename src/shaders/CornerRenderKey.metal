#ifndef CORNERRENDERKEY_MSL
#define CORNERRENDERKEY_MSL
#include "CornerNormalOffset.metal"
#include "gpu/CornerClass.h"
#include "gpu/CornerClassMode.h"
#include "gpu/MeshRecord.h"

// Shared render-vertex equivalence for finest construction and local LOD
// simplification. Callers supply canonical corner handles, including corners
// retained from different source faces in a coarse triangle.
struct CornerRenderKey {
    device const BindlessSet &B;
    MeshRecord Mesh;
    ElementAttributeRef CornerSectors;
    uint FaceSharpnessSlot;
    uint VertexIndex(uint corner) const {
        return BindlessBuffer(uint,B.IndexBuffer,Mesh.IndexSlotOffset.Slot)[corner]-Mesh.VertexOffset;
    }
    uint Face(uint corner) const {
        return BindlessBuffer(uint,B.Buffer,Mesh.Connectivity.HalfedgeFaces.Slot)[corner];
    }
    uint Class(uint c) const {
        if (Mesh.CornerClassMode == uint(CornerClassMode::UniformVertex)) return uint(CornerClass::Vertex);
        if (Mesh.CornerClassMode == uint(CornerClassMode::UniformFace)) return uint(CornerClass::Face);
        if (BindlessBuffer(uchar, B.Buffer, FaceSharpnessSlot)[Face(c)] != 0u) return uint(CornerClass::Face);
        return uint(CornerSectorRoot(B, CornerSectors, c) == InvalidOffset ? CornerClass::Vertex : CornerClass::Seam);
    }
    uint ClassIdentity(uint c, uint kind, bool flat) const {
        if (kind == uint(CornerClass::Seam)) return CornerSectorRoot(B, CornerSectors, c);
        return !flat && kind == uint(CornerClass::Face) ? Face(c) : InvalidOffset;
    }
    bool Custom(uint c) const {
        return CustomNormalOffset(B, Mesh.CustomNormals, c).x >= 0.f;
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
        const uint h = c;
        uint at = 0u;
        for (uint set = 0u; set < 4u; ++set) {
            if (Mesh.CornerUvs[set].ValuesSlot == InvalidSlot) continue;
            if (word < at + 2u) return as_type<uint>(BindlessBuffer(packed_float2, B.CornerUvBuffer, Mesh.CornerUvs[set].ValuesSlot)[ElementAttributeIndex(B, Mesh.CornerUvs[set], h)][word - at]);
            at += 2u;
        }
        if (Mesh.CornerTangent.ValuesSlot != InvalidSlot) {
            if (word < at + 4u) return as_type<uint>(BindlessBuffer(packed_float4, B.CornerTangentBuffer, Mesh.CornerTangent.ValuesSlot)[ElementAttributeIndex(B, Mesh.CornerTangent, h)][word - at]);
            at += 4u;
        }
        return as_type<uint>(BindlessBuffer(packed_float4, B.CornerColorBuffer, Mesh.CornerColor.ValuesSlot)[ElementAttributeIndex(B, Mesh.CornerColor, h)][word - at]);
    }
    uint AttributeWords() const {
        uint n = (Mesh.CornerTangent.ValuesSlot == InvalidSlot ? 0u : 4u) + (Mesh.CornerColor.ValuesSlot == InvalidSlot ? 0u : 4u);
        for (uint set = 0u; set < 4u; ++set) if (Mesh.CornerUvs[set].ValuesSlot != InvalidSlot) n += 2u;
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
