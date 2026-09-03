#pragma once

#include "gpu/CornerClass.h"
#include "gpu/CornerClassEncoding.h"
#include "numeric/vec2.h"
#include "numeric/vec4.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <span>

// Texture coordinate sets a corner carries, matching MeshStore::MaxUvSets.
constexpr uint32_t MaxWeldUvSets{4};
// The widest key: vertex id and corner class, four UV sets, tangents, and colors.
constexpr uint32_t MaxWeldKeyWords{18};

// Defines identical render-vertex equivalence for level-zero and coarse clusters.
struct CornerWeldSource {
    uint32_t CornerClassOffset{};
    std::span<const uint32_t> CornerClasses;
    std::span<const uint32_t> TriangleFaceIds;
    std::span<const uvec2> CustomCornerMasks;
    std::array<std::span<const vec2>, MaxWeldUvSets> CornerUvs;
    std::span<const vec4> CornerTangents;
    std::span<const vec4> CornerColors;
    bool MorphShadingAuthored{};
};

// Uses primitive-local corner and triangle indices relative to `first_corner`.
struct CornerWeldKey {
    CornerWeldKey(const CornerWeldSource &source, uint32_t first_corner)
        : Source(source), FirstCorner(first_corner),
          UniformClassWord(
              uint32_t(source.CornerClassOffset == uint32_t(CornerClassEncoding::UniformFaceOffset) ? CornerClass::Face : CornerClass::Vertex)
              << uint32_t(CornerClassEncoding::TagShift)
          ),
          Words(
              2u + 2u * uint32_t(std::ranges::count_if(source.CornerUvs, [](auto uvs) { return !uvs.empty(); })) +
              (source.CornerTangents.empty() ? 0u : 4u) + (source.CornerColors.empty() ? 0u : 4u)
          ) {}

    uint32_t WordCount() const { return Words; }

    // All-Face triangles omit the face ID because their primitive stores one common normal.
    bool FlatFaceTriangle(uint32_t triangle) const {
        if (Source.MorphShadingAuthored) return false;
        for (uint32_t c = 0; c < 3u; ++c) {
            const uint32_t corner = triangle * 3u + c;
            if (ClassWord(corner) >> uint32_t(CornerClassEncoding::TagShift) != uint32_t(CornerClass::Face)) return false;
            if (WeldsAlone(corner)) return false;
        }
        return true;
    }

    // A custom-normal corner's frame depends on its own triangle, so it welds with nothing.
    bool WeldsAlone(uint32_t corner) const {
        const uint32_t global = FirstCorner + corner;
        return !Source.CustomCornerMasks.empty() &&
            (Source.CustomCornerMasks[global / 32u].x & (1u << (global % 32u))) != 0u;
    }

    // Fill the first WordCount() words with the corner's key. `flat_face` marks its triangle flat.
    void Write(uint32_t corner, uint32_t source_vertex, bool flat_face, std::array<uint32_t, MaxWeldKeyWords> &words) const {
        const uint32_t global = FirstCorner + corner;
        words = {};
        words[0] = source_vertex;
        uint32_t corner_class = ClassWord(corner);
        // Preserve the source face normal in the key for Face corners outside All-Face triangles.
        if (!flat_face && corner_class >> uint32_t(CornerClassEncoding::TagShift) == uint32_t(CornerClass::Face)) {
            corner_class |= (Source.TriangleFaceIds[global / 3u] - 1u) & uint32_t(CornerClassEncoding::IndexMask);
        }
        words[1] = corner_class;
        uint32_t word = 2;
        for (const auto uvs : Source.CornerUvs) {
            if (uvs.empty()) continue;
            const vec2 uv = uvs[global];
            words[word++] = FloatBits(uv.x);
            words[word++] = FloatBits(uv.y);
        }
        if (!Source.CornerTangents.empty()) {
            const vec4 tangent = Source.CornerTangents[global];
            words[word++] = FloatBits(tangent.x);
            words[word++] = FloatBits(tangent.y);
            words[word++] = FloatBits(tangent.z);
            words[word++] = FloatBits(tangent.w);
        }
        if (!Source.CornerColors.empty()) {
            const vec4 color = Source.CornerColors[global];
            words[word++] = FloatBits(color.x);
            words[word++] = FloatBits(color.y);
            words[word++] = FloatBits(color.z);
            words[word++] = FloatBits(color.w);
        }
        assert(word == Words);
    }

private:
    static uint32_t FloatBits(float value) { return std::bit_cast<uint32_t>(value); }

    uint32_t ClassWord(uint32_t corner) const {
        return Source.CornerClasses.empty() ? UniformClassWord : Source.CornerClasses[FirstCorner + corner];
    }

    const CornerWeldSource &Source;
    uint32_t FirstCorner;
    uint32_t UniformClassWord;
    uint32_t Words;
};
