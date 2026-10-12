#pragma once

#include "gpu/MeshElementBlock.h"
#include "mesh/Mesh.h"
#include "numeric/VectorMath.h"

#include <algorithm>
#include <bit>
#include <optional>
#include <stdexcept>
#include <vector>

// Dense ordinal lookups and whole-mesh reductions for tests, over a domain's live elements in ascending handle order.
namespace test {
inline uint32_t LiveOrdinal(std::span<const uint32_t> table, std::span<const MeshElementBlock> blocks, uint32_t handle) {
    const auto local = handle / MeshElementBlockSize, slot = handle % MeshElementBlockSize;
    if (local >= table.size() || table[local] == InvalidOffset) throw std::out_of_range("The handle is outside the domain.");
    uint32_t ordinal = 0u;
    for (uint32_t l = 0u; l < local; ++l)
        if (table[l] != InvalidOffset) ordinal += blocks[table[l]].Count;
    const auto &live = blocks[table[local]].Live;
    for (uint32_t w = 0u; w < slot / 32u; ++w) ordinal += uint32_t(std::popcount(live[w]));
    return ordinal + uint32_t(std::popcount(live[slot / 32u] & ((1u << (slot % 32u)) - 1u)));
}
inline uint32_t LiveHandle(std::span<const uint32_t> table, std::span<const MeshElementBlock> blocks, uint32_t ordinal) {
    for (uint32_t local = 0u; local < table.size(); ++local) {
        if (table[local] == InvalidOffset) continue;
        const auto &block = blocks[table[local]];
        if (ordinal >= block.Count) {
            ordinal -= block.Count;
            continue;
        }
        for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) {
            auto bits = block.Live[w];
            const auto live = uint32_t(std::popcount(bits));
            if (ordinal >= live) {
                ordinal -= live;
                continue;
            }
            while (ordinal--) bits &= bits - 1u;
            return local * MeshElementBlockSize + w * 32u + uint32_t(std::countr_zero(bits));
        }
    }
    throw std::out_of_range("The ordinal exceeds the domain.");
}

inline uint32_t EdgeOrdinal(const Mesh &mesh, Mesh::EH edge) {
    const auto &c = mesh.GetConnectivity();
    return LiveOrdinal(c.EdgeTable, c.EdgeBlocks, *edge);
}
inline uint32_t HalfedgeOrdinal(const Mesh &mesh, Mesh::HH halfedge) {
    const auto &c = mesh.GetConnectivity();
    return LiveOrdinal(c.HalfedgeTable, c.HalfedgeBlocks, *halfedge);
}
inline Mesh::HH HalfedgeAt(const Mesh &mesh, uint32_t ordinal) {
    const auto &c = mesh.GetConnectivity();
    return Mesh::HH{LiveHandle(c.HalfedgeTable, c.HalfedgeBlocks, ordinal)};
}
inline he::HandleRange<Mesh::HH> Halfedges(const Mesh &mesh) {
    const auto &c = mesh.GetConnectivity();
    return {c.HalfedgeTable, c.HalfedgeBlocks};
}
// The arena address of a mesh's element, which indexes its arena values.
inline uint32_t Address(const Mesh &mesh, Mesh::VH v) { return mesh.GetConnectivity().VertexAddress(*v).Value(); }
inline uint32_t Address(const Mesh &mesh, Mesh::HH h) { return mesh.GetConnectivity().HalfedgeAddress(*h).Value(); }
inline uint32_t Address(const Mesh &mesh, Mesh::EH e) { return mesh.GetConnectivity().EdgeAddress(*e).Value(); }
inline uint32_t Address(const Mesh &mesh, Mesh::FH f) { return mesh.GetConnectivity().FaceAddress(*f).Value(); }
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
