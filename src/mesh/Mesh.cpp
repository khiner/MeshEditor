#include "numeric/VectorMath.h"

#include "GeometrySelection.h"
#include "Mesh.h"

#include "MeshComponents.h"
#include "MeshStore.h"

#include "state/Scene.h"

using numeric::dvec3;

using std::ranges::distance;

namespace {
// Consume the stored tessellation.
// Polygon-loop order does not define triangles.
void ForEachFaceTriangle(const Mesh &mesh, auto &&fn) {
    for (const auto corners : mesh.DerivedTriangles())
        fn(*mesh.GetToVertex(Mesh::HH{corners.x}), *mesh.GetToVertex(Mesh::HH{corners.y}), *mesh.GetToVertex(Mesh::HH{corners.z}));
}
} // namespace

// By Ericson's region test: the nearest point lies in the interior, on an edge, or at a vertex.
// Each region is decided by a pair of dot products.
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
    if (sum <= 0) return {a, {1, 0, 0}}; // Degenerate triangle, whose interior the region tests cannot reach.
    const float v = vb / sum, w = vc / sum;
    return {a + ab * v + ac * w, {1 - v - w, v, w}};
}

Mesh::Mesh(const MeshStore &store, uint32_t store_id)
    : Store(&store), StoreId(store_id), C(store.GetConnectivity(store_id)), Corners(store.Arenas().FaceCorners.Buffer.GetSpan<uint32_t>()) {}

Mesh::VH Mesh::VertexAt(uint32_t ordinal) const { return VH{Store->LiveElementAt(StoreId, MeshStore::ElementDomain::Vertex, ordinal)}; }
uint32_t Mesh::VertexOrdinal(VH vertex) const { return Store->LiveElementOrdinal(StoreId, MeshStore::ElementDomain::Vertex, *vertex); }
Mesh::EH Mesh::EdgeAt(uint32_t ordinal) const { return EH{Store->LiveElementAt(StoreId, MeshStore::ElementDomain::Edge, ordinal)}; }
Mesh::FH Mesh::FaceAt(uint32_t ordinal) const { return FH{Store->LiveElementAt(StoreId, MeshStore::ElementDomain::Face, ordinal)}; }
uint32_t Mesh::FaceOrdinal(FH face) const { return Store->LiveElementOrdinal(StoreId, MeshStore::ElementDomain::Face, *face); }

std::span<const uint32_t> Mesh::CornerVertices() const { return Store->Arenas().FaceCorners.Get(Store->Get(StoreId).FaceCorners); }

Mesh GetMesh(const state::Scene &r, state::Entity e) {
    return {r.Context.get<const MeshStore>(), r.get<const MeshHandle>(e).StoreId};
}
std::optional<Mesh> TryGetMesh(const state::Scene &r, state::Entity e) {
    if (!HasMesh(r, e)) return std::nullopt;
    return GetMesh(r, e);
}
bool HasMesh(const state::Scene &r, state::Entity e) { return r.all_of<MeshHandle>(e); }
std::optional<uint32_t> DrawnStoreId(const state::Scene &r, state::Entity e) {
    if (HasMesh(r, e)) return r.get<const MeshHandle>(e).StoreId;
    if (const auto *vertices = r.try_get<const VertexStoreId>(e)) return vertices->StoreId;
    return std::nullopt;
}

he::VH Mesh::GetFromVertex(HH hh) const {
    assert(*hh < C.Opposites.size());
    if (const auto opp = C.Opposites[*hh]) return VH(Corners[*opp]);
    // A boundary halfedge has no opposite, so its from-vertex comes from the previous halfedge in the face loop.
    const auto prev = C.Previous(hh);
    return prev ? VH(Corners[*prev]) : VH{};
}

uint32_t Mesh::GetValence(FH fh) const { return distance(fh_range(fh)); }

vec3 Mesh::CalcFaceCentroid(FH fh) const {
    assert(FaceOrdinal(fh) < C.FaceCount);
    vec3 centroid{0};
    uint32_t count{0};
    for (auto vh : fv_range(fh)) {
        centroid += GetPosition(vh);
        count++;
    }
    return count > 0 ? centroid / float(count) : centroid;
}

float Mesh::CalcMeanCurvature(VH vh, std::span<const uint8_t> edge_sharpness) const {
    for (const auto he : voh_range(vh)) {
        const auto eh = GetEdge(he);
        if (*eh < edge_sharpness.size() && edge_sharpness[*eh] != 0) return 0.f;
    }

    const vec3 xi = GetPosition(vh);
    const vec3 ni = Normalize(GetNormal(vh));
    double sum = 0;
    int count = 0;
    for (const auto he : voh_range(vh)) {
        // A halfedge with no opposite bounds the surface rather than running through it.
        if (!GetOppositeHalfedge(he)) continue;
        const vec3 d = GetPosition(GetToVertex(he)) - xi;
        const double d2 = Dot(d, d);
        if (d2 < 1e-20) continue;
        sum += -2.0 * double(Dot(d, ni)) / d2;
        ++count;
    }
    return count ? float(sum / count) : 0.f;
}

he::VH Mesh::FindNearestVertex(vec3 p) const {
    VH closest_vertex;
    float min_dist_sq = std::numeric_limits<float>::max();
    for (const auto vh : vertices()) {
        if (const float dist_sq = Distance2(GetPosition(vh), p); dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            closest_vertex = vh;
        }
    }
    return closest_vertex;
}

const vec3 &Mesh::GetPosition(VH vh) const { return Store->Arenas().Vertices.Get({*vh, 1})[0].Position; }
const vec3 &Mesh::GetNormal(VH vh) const { return Store->Arenas().BaseVertexNormals.Get({*vh, 1})[0]; }
vec3 Mesh::GetNormal(FH fh) const { return Store->Arenas().BaseFaceNormals.Get({*fh, 1})[0]; }
VertexEdgeIncidence Mesh::GetVertexEdgeIncidence() const { return Store->GetVertexEdgeIncidence(StoreId); }

ElementView<uvec3> Mesh::DerivedTriangles() const { return Store->TriangleView(StoreId); }

TriangleVertexView Mesh::TriangleVertices() const {
    if (!TriangleIndexCount()) return {};
    return {Store->GetTriangleCorners(StoreId), Store->Arenas().FaceCorners.Buffer.GetSpan<uint32_t>()};
}

uint32_t Mesh::TriangleIndexCount() const { return Store->Get(StoreId).TriangleCount * 3; }

void Mesh::WriteTriangleIndices(std::span<uint32_t> dest) const {
    uint32_t i = 0;
    ForEachFaceTriangle(*this, [&](uint32_t v0, uint32_t v1, uint32_t v2) {
        dest[i++] = VertexOrdinal(VH{v0});
        dest[i++] = VertexOrdinal(VH{v1});
        dest[i++] = VertexOrdinal(VH{v2});
    });
}

std::vector<uint32_t> Mesh::CreateTriangleIndices() const {
    std::vector<uint32_t> indices(TriangleIndexCount());
    WriteTriangleIndices(indices);
    return indices;
}

VertexEdgeIncidence::Iterator &VertexEdgeIncidence::Iterator::operator++() {
    Edge = he::null;
    while (Remaining) {
        const auto h = he::HH{C->FanItems[Item].x};
        if (Side++ == 0) {
            const auto e = C->Edge(h);
            const auto first = C->EdgeHalfedge(*e);
            if (h == first || h == C->Opposites[*first]) {
                Edge = *e;
                return *this;
            }
        } else {
            const auto next = C->Next(h);
            ++Item;
            --Remaining;
            Side = 0;
            if (next && !C->Opposites[*next] && C->EdgeHalfedge(*C->Edge(next)) == next) {
                Edge = *C->Edge(next);
                return *this;
            }
        }
    }
    Side = 0;
    return *this;
}

void Mesh::ValidateSelection(const GeometrySelection &selection) const {
    if (!Store) throw std::invalid_argument("Geometry planning requires a bound mesh.");
    ValidateGeometrySelection(*Store, StoreId, selection);
}

void ValidateGeometrySelection(const MeshStore &meshes, uint32_t id, const GeometrySelection &selection) {
    if (!meshes.TryGet(id)) throw std::invalid_argument("Geometry selection requires a live mesh.");
    for (const auto element : he::Elements) {
        const auto &handles = selection.Get(element);
        if (!std::ranges::is_sorted(handles) || std::ranges::adjacent_find(handles) != handles.end())
            throw std::invalid_argument("Geometry selections require sorted unique handles.");
        for (const auto handle : handles)
            if (!meshes.IsLiveElement(id, element, handle))
                throw std::invalid_argument("Geometry selection handle is not owned by its mesh.");
    }
}
