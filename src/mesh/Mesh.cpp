#include "Mesh.h"

#include "MeshComponents.h"
#include "MeshStore.h"
#include "procedure/Procedure.h"

#include "state/Registry.h"

// Ericson's region test, where a pair of dot products decides each vertex, edge or interior region.
TrianglePoint ClosestPointOnTriangle(vec3 p, vec3 a, vec3 b, vec3 c) {
    const vec3 ab = b - a, ac = c - a;
    const vec3 ap = p - a, bp = p - b, cp = p - c;
    const float d1 = Dot(ab, ap), d2 = Dot(ac, ap);
    const float d3 = Dot(ab, bp), d4 = Dot(ac, bp);
    const float d5 = Dot(ab, cp), d6 = Dot(ac, cp);

    if (d1 <= 0 && d2 <= 0) return {a, {1, 0, 0}};
    if (d3 >= 0 && d4 <= d3) return {b, {0, 1, 0}};
    if (d6 >= 0 && d5 <= d6) return {c, {0, 0, 1}};

    const float va = d3 * d6 - d5 * d4, vb = d5 * d2 - d1 * d6, vc = d1 * d4 - d3 * d2;
    // Zero-length edges defer to a nonzero edge.
    // A point-degenerate triangle reaches the fallback below.
    const float ab_len2 = d1 - d3, ac_len2 = d2 - d6, bc_len2 = (d4 - d3) + (d5 - d6);
    if (vc <= 0 && d1 >= 0 && d3 <= 0 && ab_len2 > 0) {
        const float v = d1 / ab_len2;
        return {a + ab * v, {1 - v, v, 0}};
    }
    if (vb <= 0 && d2 >= 0 && d6 <= 0 && ac_len2 > 0) {
        const float w = d2 / ac_len2;
        return {a + ac * w, {1 - w, 0, w}};
    }
    if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0 && bc_len2 > 0) {
        const float w = (d4 - d3) / bc_len2;
        return {b + (c - b) * w, {0, 1 - w, w}};
    }

    const float sum = va + vb + vc;
    if (sum <= 0) return {a, {1, 0, 0}};
    const float v = vb / sum, w = vc / sum;
    return {a + ab * v + ac * w, {1 - v - w, v, w}};
}

Mesh::Mesh(const MeshStore &store, uint32_t store_id)
    : Store(&store), StoreId(store_id), C(store.GetConnectivity(store_id)) {}

Mesh::VH Mesh::VertexAt(uint32_t ordinal) const { return VH{Store->LiveElementAt(StoreId, MeshStore::ElementDomain::Vertex, ordinal).Value}; }
uint32_t Mesh::VertexOrdinal(VH vertex) const { return Store->LiveElementOrdinal(StoreId, MeshStore::ElementDomain::Vertex, ElementHandle{*vertex}); }
Mesh::EH Mesh::EdgeAt(uint32_t ordinal) const { return EH{Store->LiveElementAt(StoreId, MeshStore::ElementDomain::Edge, ordinal).Value}; }
Mesh::FH Mesh::FaceAt(uint32_t ordinal) const { return FH{Store->LiveElementAt(StoreId, MeshStore::ElementDomain::Face, ordinal).Value}; }
uint32_t Mesh::FaceOrdinal(FH face) const { return Store->LiveElementOrdinal(StoreId, MeshStore::ElementDomain::Face, ElementHandle{*face}); }

bool HasMesh(const state::Registry &r, state::Entity e) { return r.all_of<procedure::GeometryProcedure>(e); }
bool DrawsMesh(const state::Registry &r, state::Entity e) { return r.any_of<EvaluatedMesh, MeshHandle>(e); }
std::optional<uint32_t> DrawnStoreId(const state::Registry &r, state::Entity e) {
    if (const auto *evaluated = r.try_get<const EvaluatedMesh>(e)) return evaluated->StoreId;
    if (const auto *handle = r.try_get<const MeshHandle>(e)) return handle->StoreId;
    if (const auto *vertices = r.try_get<const VertexStoreId>(e)) return vertices->StoreId;
    return std::nullopt;
}
Mesh GetMesh(const state::Registry &r, state::Entity e) { return {r.Resources.get<const MeshStore>(), *DrawnStoreId(r, e)}; }
std::optional<Mesh> TryGetMesh(const state::Registry &r, state::Entity e) {
    if (!DrawsMesh(r, e)) return std::nullopt;
    return GetMesh(r, e);
}

ElementView<uvec3> Mesh::DerivedTriangles() const { return Store->TriangleView(StoreId); }

HandleValues<Vertex> Mesh::VertexValues() const { return {Store->Arenas().Vertices.Buffer.GetSpan<Vertex>(), C.VertexTable}; }

TriangleVertexView Mesh::TriangleVertices() const {
    if (!TriangleIndexCount()) return {};
    return {Store->GetTriangleCorners(StoreId), Store->Arenas().FaceCorners.Buffer.GetSpan<uint32_t>()};
}

uint32_t Mesh::TriangleIndexCount() const { return Store->Get(StoreId).TriangleCount * 3; }

GeometrySelection AllHandles(const Mesh &mesh) { return {HandleList(mesh.vertices()), HandleList(mesh.edges()), HandleList(mesh.faces())}; }

SlotOffset MaskBinding(SelectionMasks masks) { return {masks.Arena->Buffer.Slot, masks.First * MeshElementBlockWords}; }

std::vector<uint32_t> ReadMasks(const MeshStore &meshes, uint32_t id, Element element, SelectionMasks masks) {
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    const auto words = masks.Arena->Buffer.GetSpan<uint32_t>();
    std::vector<uint32_t> handles;
    const auto read = [&](const auto &arena, ElementSetRef set) {
        arena.ForEachBlock(set, [&](uint32_t b, const auto &block) {
            const auto mask = words.subspan(uint64_t(masks.First + b) * MeshElementBlockWords, MeshElementBlockWords);
            for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) {
                for (auto bits = block.Live[w] & mask[w]; bits; bits &= bits - 1u) handles.push_back(block.Local * MeshElementBlockSize + w * 32u + uint32_t(std::countr_zero(bits)));
            }
        });
    };
    if (element == Element::Vertex) read(a.Vertices, record.Vertices);
    else if (element == Element::Edge) read(a.EdgeHalfedges, record.EdgeData);
    else read(a.FaceTriangles, record.FaceData);
    return handles;
}

GeometrySelection ReadMasks(const MeshStore &meshes, uint32_t id, const GeometryMasks &masks) {
    return {ReadMasks(meshes, id, Element::Vertex, masks.Vertices), ReadMasks(meshes, id, Element::Edge, masks.Edges), ReadMasks(meshes, id, Element::Face, masks.Faces)};
}

SelectedHandles::SelectedHandles(const MeshStore &meshes, uint32_t id, const SelectionSource &source) : Lists{std::get_if<GeometrySelection>(&source)} {
    if (const auto *masks = std::get_if<GeometryMasks>(&source)) Read = ReadMasks(meshes, id, *masks);
}
