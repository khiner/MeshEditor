#pragma once

#include "gpu/MeshElementBlock.h"
#include "mesh/Mesh.h"
#include "numeric/VectorMath.h"

#include <algorithm>
#include <bit>
#include <optional>
#include <stdexcept>
#include <vector>

// Dense ordinal lookups and whole-mesh reductions for tests, over a domain's live blocks in ascending handle order.
namespace test {
inline std::vector<uint32_t> SortedBlocks(std::span<const MeshElementBlock> blocks, uint32_t first_handle, uint32_t count) {
    std::vector<uint32_t> sorted;
    if (!count) return sorted;
    for (auto b = first_handle / MeshElementBlockSize; b != InvalidOffset; b = blocks[b].Next) sorted.push_back(b);
    std::ranges::sort(sorted);
    return sorted;
}
inline uint32_t LiveOrdinal(std::span<const MeshElementBlock> blocks, uint32_t first_handle, uint32_t count, uint32_t handle) {
    const auto block = handle / MeshElementBlockSize, slot = handle % MeshElementBlockSize;
    uint32_t ordinal = 0u;
    bool owned = false;
    for (auto b = count ? first_handle / MeshElementBlockSize : InvalidOffset; b != InvalidOffset; b = blocks[b].Next) {
        if (b < block) ordinal += blocks[b].Count;
        owned = owned || b == block;
    }
    if (!owned) throw std::out_of_range("The handle is outside the domain.");
    for (uint32_t w = 0u; w < slot / 32u; ++w) ordinal += uint32_t(std::popcount(blocks[block].Live[w]));
    return ordinal + uint32_t(std::popcount(blocks[block].Live[slot / 32u] & ((1u << (slot % 32u)) - 1u)));
}
inline uint32_t LiveHandle(std::span<const MeshElementBlock> blocks, uint32_t first_handle, uint32_t count, uint32_t ordinal) {
    for (const auto b : SortedBlocks(blocks, first_handle, count)) {
        if (ordinal >= blocks[b].Count) {
            ordinal -= blocks[b].Count;
            continue;
        }
        for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) {
            auto bits = blocks[b].Live[w];
            const auto live = uint32_t(std::popcount(bits));
            if (ordinal >= live) {
                ordinal -= live;
                continue;
            }
            while (ordinal--) bits &= bits - 1u;
            return b * MeshElementBlockSize + w * 32u + uint32_t(std::countr_zero(bits));
        }
    }
    throw std::out_of_range("The ordinal exceeds the domain.");
}

inline uint32_t EdgeOrdinal(const Mesh &mesh, Mesh::EH edge) {
    return LiveOrdinal(mesh.GetConnectivity().EdgeBlocks, mesh.EdgeFirst(), mesh.EdgeCount(), *edge);
}
inline uint32_t HalfedgeOrdinal(const Mesh &mesh, Mesh::HH halfedge) {
    return LiveOrdinal(mesh.GetConnectivity().HalfedgeBlocks, mesh.HalfedgeFirst(), mesh.HalfEdgeCount(), *halfedge);
}
inline Mesh::HH HalfedgeAt(const Mesh &mesh, uint32_t ordinal) {
    return Mesh::HH{LiveHandle(mesh.GetConnectivity().HalfedgeBlocks, mesh.HalfedgeFirst(), mesh.HalfEdgeCount(), ordinal)};
}
inline he::HandleRange<Mesh::HH> Halfedges(const Mesh &mesh) {
    return {mesh.GetConnectivity().HalfedgeBlocks, mesh.HalfEdgeCount() ? mesh.HalfedgeFirst() / MeshElementBlockSize : he::null};
}
// Two vertex ordinals per edge in edge ordinal order, zero for a face-less halfedge's missing endpoint.
inline void WriteEdgeIndices(const Mesh &mesh, std::span<uint32_t> dest) {
    uint32_t i = 0u;
    for (uint32_t e = 0u; e < mesh.EdgeCount(); ++e) {
        const auto h = mesh.GetHalfedge(mesh.EdgeAt(e), 0);
        const auto from = mesh.GetFromVertex(h), to = mesh.GetToVertex(h);
        dest[i++] = from && to ? mesh.VertexOrdinal(from) : 0u;
        dest[i++] = from && to ? mesh.VertexOrdinal(to) : 0u;
    }
}
// The volume a closed manifold surface encloses, from the signed tetrahedra its triangles span with the origin.
inline std::optional<double> EnclosedVolume(const Mesh &mesh) {
    if (!mesh.HasClosedSurface()) return std::nullopt;
    double volume = 0.0;
    for (const auto corners : mesh.DerivedTriangles()) {
        const numeric::dvec3 a{mesh.GetPosition(mesh.GetToVertex(Mesh::HH{corners.x}))}, b{mesh.GetPosition(mesh.GetToVertex(Mesh::HH{corners.y}))},
            c{mesh.GetPosition(mesh.GetToVertex(Mesh::HH{corners.z}))};
        volume += Dot(a, Cross(b, c)) / 6.0;
    }
    return std::abs(volume);
}
} // namespace test
