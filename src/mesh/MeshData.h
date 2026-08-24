#pragma once

#include "numeric/vec3.h"

#include <span>

struct MeshData {
    MeshData() = default;
    explicit MeshData(std::vector<vec3> positions) : Positions{std::move(positions)} {}
    MeshData(std::vector<vec3> positions, const std::vector<std::vector<uint32_t>> &loops) : Positions{std::move(positions)} {
        uint32_t corner_count = 0;
        for (const auto &loop : loops) corner_count += uint32_t(loop.size());
        FaceOffsets.reserve(loops.size() + 1);
        FaceCorners.reserve(corner_count);
        for (const auto &loop : loops) AddFace(loop);
    }

    std::vector<vec3> Positions;
    // Face vertex-index loops (triangles/polygons) concatenated: face `f` spans
    // [FaceOffsets[f], FaceOffsets[f + 1]) of FaceCorners, and FaceOffsets leads with a zero.
    std::vector<uint32_t> FaceOffsets{}, FaceCorners{};
    std::vector<std::array<uint32_t, 2>> Edges{}; // Line segment vertex index pairs

    uint32_t FaceCount() const { return FaceOffsets.size() < 2u ? 0u : uint32_t(FaceOffsets.size() - 1u); }
    uint32_t FaceSize(uint32_t face) const { return FaceOffsets[face + 1] - FaceOffsets[face]; }
    std::span<const uint32_t> Face(uint32_t face) const {
        return std::span{FaceCorners}.subspan(FaceOffsets[face], FaceSize(face));
    }

    void AddFace(std::span<const uint32_t> loop) {
        if (FaceOffsets.empty()) FaceOffsets.emplace_back(0u);
        FaceCorners.insert(FaceCorners.end(), loop.begin(), loop.end());
        FaceOffsets.emplace_back(uint32_t(FaceCorners.size()));
    }
    // Reserve for `faces` more faces of `corners_per_face` corners each.
    void ReserveFaces(uint32_t faces, uint32_t corners_per_face) {
        if (FaceOffsets.empty()) FaceOffsets.emplace_back(0u);
        FaceOffsets.reserve(FaceOffsets.size() + faces);
        FaceCorners.reserve(FaceCorners.size() + size_t(faces) * corners_per_face);
    }

    // Flatten edge pairs into a contiguous index array for GPU line rendering.
    std::vector<uint32_t> CreateEdgeIndices() const {
        std::vector<uint32_t> indices;
        indices.reserve(Edges.size() * 2);
        for (const auto &[a, b] : Edges) {
            indices.emplace_back(a);
            indices.emplace_back(b);
        }
        return indices;
    }
};
