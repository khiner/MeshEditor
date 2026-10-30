#pragma once

#include "mesh/Mesh.h"
#include "numeric/VectorMath.h"

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
inline uint32_t EdgeOrdinal(const Mesh &mesh, Mesh::EH edge) {
    const auto &c = mesh.GetConnectivity();
    return LiveOrdinal(c.EdgeTable, c.EdgeBlocks, *edge);
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
