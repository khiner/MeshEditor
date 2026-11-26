#pragma once

#include "gpu/CornerClass.h"
#include "gpu/MeshAttributeBit.h"

// Texture coordinate sets a corner carries, matching MeshStore::MaxUvSets.
constexpr uint32_t MaxWeldUvSets{4};
// Vertex, class, face/sector identity, custom-normal corner identity, four UV sets, tangent, and color.
constexpr uint32_t MaxWeldKeyWords{20};

// Defines identical render-vertex equivalence for level-zero and coarse clusters.
struct CornerWeldSource {
    uint32_t CornerClassMode{InvalidOffset};
    CornerAttributeView<uint32_t> CornerSectors;
    std::span<const uint8_t> FaceSharpness;
    std::span<const uint32_t> FaceTable; // Resolves face handles to FaceSharpness addresses, and an empty table indexes it by handle.
    TriangleFaceView TriangleFaces;
    CornerAttributeView<CustomNormal> CustomNormals;
    std::array<CornerAttributeView<vec2>, MaxWeldUvSets> CornerUvs;
    CornerAttributeView<vec4> CornerTangents;
    CornerAttributeView<vec4> CornerColors;
    bool MorphShadingAuthored{};
};

// Draw-corner input resolves once to canonical handles.
// Meshlets use handles directly.
// `attributes` holds the MeshAttributeBit channels a primitive's coarse levels preserve, and the key covers its texture coordinate sets and tangents among them.
struct CornerWeldKey {
    CornerWeldKey(const CornerWeldSource &source, uint32_t first_corner, uint32_t attributes)
        : Uvs{Keyed(source, attributes, 0u), Keyed(source, attributes, 1u), Keyed(source, attributes, 2u), Keyed(source, attributes, 3u)},
          Tangents((attributes & MeshAttributeBit_Tangent) && !source.CornerTangents.empty()),
          Source(source), FirstCorner(first_corner),
          Words(3u + (source.CustomNormals.empty() ? 0u : 1u) + 2u * uint32_t(std::ranges::count(Uvs, true)) + (Tangents ? 4u : 0u) + (source.CornerColors.empty() ? 0u : 4u)) {}

    // The texture coordinate sets and the tangents the key covers.
    const std::array<bool, MaxWeldUvSets> Uvs;
    const bool Tangents;

    uint32_t WordCount() const { return Words; }

    // All-Face triangles omit the face ID because their primitive stores one common normal.
    bool FlatFaceTriangle(uint32_t triangle) const {
        if (Source.MorphShadingAuthored) return false;
        for (uint32_t c = 0; c < 3u; ++c) {
            const uint32_t corner = triangle * 3u + c;
            if (ClassWord(corner) != uint32_t(CornerClass::Face)) return false;
            if (HasCustomNormal(corner)) return false;
        }
        return true;
    }

    // The arena address of a corner handle, which indexes the corners' faces, while the corner layers index the handle.
    uint32_t Address(uint32_t handle) const {
        const auto &table = Source.TriangleFaces.Corners.CornerTable;
        return table.empty() ? handle : Resolve<MeshElementBlockSize>(table, ElementHandle{handle}).Value();
    }

    uint32_t Handle(uint32_t corner) const {
        const auto global = FirstCorner + corner;
        return Source.TriangleFaces.Corners.Values.empty() ? global : Source.TriangleFaces.Corners[global];
    }

    // Distinct polygon frames may decode the same offsets differently.
    bool HasCustomNormal(uint32_t corner) const {
        return Source.CustomNormals.Attribute.GetOr(Handle(corner)).Offset.x >= 0.f;
    }

    // Fill the first WordCount() words with the corner's key.
    // `flat_face` marks its triangle flat.
    void Write(uint32_t corner, uint32_t source_vertex, bool flat_face, std::array<uint32_t, MaxWeldKeyWords> &words) const {
        WriteHandle(Handle(corner), source_vertex, flat_face, words);
    }

    void WriteHandle(uint32_t handle, uint32_t source_vertex, bool flat_face, std::array<uint32_t, MaxWeldKeyWords> &words) const {
        words = {};
        words[0] = source_vertex;
        const uint32_t corner_class = ClassHandle(handle);
        words[2] = InvalidOffset;
        // Face identity must retain all 32 bits independently of the class tag.
        if (!flat_face && corner_class == uint32_t(CornerClass::Face)) {
            words[2] = Source.TriangleFaces.HalfedgeFaces[Address(handle)];
        }
        if (corner_class == uint32_t(CornerClass::Seam)) words[2] = Source.CornerSectors.Attribute.GetOr(handle, InvalidOffset);
        words[1] = corner_class;
        uint32_t word = 3;
        if (!Source.CustomNormals.empty()) words[word++] = Source.CustomNormals.Attribute.GetOr(handle).Offset.x >= 0.f ? handle : InvalidOffset;
        for (uint32_t set = 0; set < MaxWeldUvSets; ++set) {
            if (!Uvs[set]) continue;
            const vec2 uv = Source.CornerUvs[set].Attribute[handle];
            words[word++] = FloatBits(uv.x);
            words[word++] = FloatBits(uv.y);
        }
        if (Tangents) {
            const vec4 tangent = Source.CornerTangents.Attribute[handle];
            words[word++] = FloatBits(tangent.x);
            words[word++] = FloatBits(tangent.y);
            words[word++] = FloatBits(tangent.z);
            words[word++] = FloatBits(tangent.w);
        }
        if (!Source.CornerColors.empty()) {
            const vec4 color = Source.CornerColors.Attribute[handle];
            words[word++] = FloatBits(color.x);
            words[word++] = FloatBits(color.y);
            words[word++] = FloatBits(color.z);
            words[word++] = FloatBits(color.w);
        }
        assert(word == Words);
    }

private:
    static uint32_t FloatBits(float value) { return std::bit_cast<uint32_t>(value); }

    static bool Keyed(const CornerWeldSource &source, uint32_t attributes, uint32_t set) {
        return (attributes & (uint32_t(MeshAttributeBit_TexCoord0) << set)) && !source.CornerUvs[set].empty();
    }

    uint32_t ClassWord(uint32_t corner) const { return ClassHandle(Handle(corner)); }

    uint32_t ClassHandle(uint32_t handle) const {
        if (Source.CornerClassMode == uint32_t(CornerClassMode::UniformFace)) return uint32_t(CornerClass::Face);
        if (Source.CornerClassMode == uint32_t(CornerClassMode::UniformVertex)) return uint32_t(CornerClass::Vertex);
        const auto face = Source.TriangleFaces.HalfedgeFaces[Address(handle)];
        if (Source.FaceSharpness[Source.FaceTable.empty() ? face : Resolve<MeshElementBlockSize>(Source.FaceTable, ElementHandle{face}).Value()]) return uint32_t(CornerClass::Face);
        return uint32_t(Source.CornerSectors.Attribute.GetOr(handle, InvalidOffset) == InvalidOffset ? CornerClass::Vertex : CornerClass::Seam);
    }

    const CornerWeldSource &Source;
    uint32_t FirstCorner;
    uint32_t Words;
};
