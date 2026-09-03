#pragma once

#include "numeric/vec3.h"

#include <array>
#include <span>
#include <vector>

struct MeshData {
    MeshData() = default;
    explicit MeshData(std::vector<vec3> positions) : Positions{std::move(positions)} {}
    MeshData(std::vector<vec3> positions, const std::vector<std::vector<uint32_t>> &loops) : Positions{std::move(positions)} {
        uint32_t corner_count = 0;
        for (const auto &loop : loops) corner_count += uint32_t(loop.size());
        FaceCorners.reserve(corner_count);
        for (const auto &loop : loops) AddFace(loop);
    }

    std::vector<vec3> Positions;
    // FaceOffsets stays empty for all-triangle meshes and otherwise delimits concatenated FaceCorners loops.
    std::vector<uint32_t> FaceOffsets{}, FaceCorners{};
    std::vector<std::array<uint32_t, 2>> Edges{};

    uint32_t FaceCount() const { return Faces; }
    uint32_t FaceStart(uint32_t face) const { return FaceOffsets.empty() ? 3u * face : FaceOffsets[face]; }
    uint32_t FaceSize(uint32_t face) const { return FaceOffsets.empty() ? 3u : FaceOffsets[face + 1] - FaceOffsets[face]; }
    std::span<const uint32_t> Face(uint32_t face) const {
        return std::span{FaceCorners}.subspan(FaceStart(face), FaceSize(face));
    }

    void AddFace(std::span<const uint32_t> loop) {
        if (loop.size() != 3 && FaceOffsets.empty()) SpellOutFaceOffsets();
        FaceCorners.insert(FaceCorners.end(), loop.begin(), loop.end());
        if (!FaceOffsets.empty()) FaceOffsets.emplace_back(uint32_t(FaceCorners.size()));
        ++Faces;
    }
    // Appends `corner_count` triangle corners and returns their writable span.
    std::span<uint32_t> AddTriangleCorners(uint32_t corner_count) {
        const auto first = uint32_t(FaceCorners.size());
        FaceCorners.resize(first + corner_count);
        if (!FaceOffsets.empty()) {
            for (uint32_t c = first + 3u; c <= first + corner_count; c += 3u) FaceOffsets.emplace_back(c);
        }
        Faces += corner_count / 3u;
        return std::span{FaceCorners}.subspan(first, corner_count);
    }
    void ReserveFaces(uint32_t faces, uint32_t corners_per_face) {
        FaceCorners.reserve(FaceCorners.size() + size_t(faces) * corners_per_face);
        if (!FaceOffsets.empty()) FaceOffsets.reserve(FaceOffsets.size() + faces);
    }

private:
    uint32_t Faces{0};

    // Materialize the offsets a mesh of triangles computes arithmetically, so a face of another size can follow.
    void SpellOutFaceOffsets() {
        FaceOffsets.resize(size_t(Faces) + 1);
        for (uint32_t f = 0; f <= Faces; ++f) FaceOffsets[f] = 3u * f;
    }
};
