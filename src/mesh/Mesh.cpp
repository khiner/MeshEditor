#include "numeric/VectorMath.h"
#include "numeric/dvec3.h"
#include "numeric/vec2.h"

#include "Mesh.h"

#include "MeshComponents.h"
#include "MeshStore.h"

#include "state/Scene.h"

#include <algorithm>

using numeric::dvec3;

using std::ranges::distance;

namespace {
// Calls `fn(v0, v1, v2)` for each triangle of each face's fan, in face order.
void ForEachFaceTriangle(const Mesh &mesh, auto &&fn) {
    const auto &c = mesh.GetConnectivity();
    const auto corners = mesh.CornerVertices();
    for (uint32_t face = 0; face < c.FaceCount; ++face) {
        const auto first = *c.FaceHalfedge(face), last = c.FaceEnd(face);
        for (auto h = first + 1; h + 1 < last; ++h) fn(corners[first], corners[h], corners[h + 1]);
    }
}
} // namespace

uint32_t BuildConnectivity(std::span<const std::array<uint32_t, 2>> edge_pairs, uint32_t vertex_count, const ConnectivityStorage &storage) {
    auto outgoing = storage.OutgoingHalfedges.first(vertex_count);
    std::ranges::fill(outgoing, he::HH{});
    const auto halfedge_count = uint32_t(edge_pairs.size()) * 2u;
    auto opposites = storage.Opposites.first(halfedge_count);
    for (uint32_t e = 0; e < edge_pairs.size(); ++e) {
        const auto [a, b] = edge_pairs[e];
        const auto h0 = he::HH(e * 2u), h1 = he::HH(e * 2u + 1u);
        opposites[*h0] = h1;
        opposites[*h1] = h0;
        storage.Edges[e] = h0;
        storage.HalfedgeToEdge[*h0] = storage.HalfedgeToEdge[*h1] = he::EH(e);
        if (!outgoing[a]) outgoing[a] = h0;
        if (!outgoing[b]) outgoing[b] = h1;
    }
    return uint32_t(edge_pairs.size());
}

Mesh::Mesh(const MeshStore &store, uint32_t store_id)
    : Store(&store), StoreId(store_id), C(store.GetConnectivity(store_id)), Corners(store.Arenas().FaceCorners.Get(store.Get(store_id).FaceCorners)) {}

namespace {
uint32_t MeshStoreId(const state::Scene &r, state::Entity e) {
    if (const auto *preview = r.try_get<const MeshPreview>(e)) return preview->StoreId;
    return r.get<const MeshHandle>(e).StoreId;
}
} // namespace

Mesh GetMesh(const state::Scene &r, state::Entity e) {
    return {r.Context.get<const MeshStore>(), MeshStoreId(r, e)};
}
std::optional<Mesh> TryGetMesh(const state::Scene &r, state::Entity e) {
    if (!HasMesh(r, e)) return std::nullopt;
    return GetMesh(r, e);
}
bool HasMesh(const state::Scene &r, state::Entity e) { return r.all_of<MeshHandle>(e); }
std::optional<uint32_t> DrawnStoreId(const state::Scene &r, state::Entity e) {
    if (HasMesh(r, e)) return MeshStoreId(r, e);
    if (const auto *vertices = r.try_get<const VertexStoreId>(e)) return vertices->StoreId;
    return std::nullopt;
}

float LocalLengthPerUv(const state::Scene &r, state::Entity mesh_entity, uint32_t uv_set) {
    const auto mesh = TryGetMesh(r, mesh_entity);
    if (!mesh || uv_set >= MeshStore::MaxUvSets) return 0;
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto uvs = meshes.Arenas().CornerUvs.Get(meshes.Get(mesh->GetStoreId()).CornerUvs[uv_set]);
    const auto corners = mesh->CreateTriangleIndices();
    if (uvs.size() != corners.size() || corners.empty()) return 0;

    double world_area = 0, uv_area = 0;
    for (size_t t = 0; t + 2 < corners.size(); t += 3) {
        const vec3 p0 = mesh->GetPosition(Mesh::VH{corners[t]});
        const vec3 p1 = mesh->GetPosition(Mesh::VH{corners[t + 1]});
        const vec3 p2 = mesh->GetPosition(Mesh::VH{corners[t + 2]});
        world_area += 0.5 * double(Length(Cross(p1 - p0, p2 - p0)));
        const vec2 a = uvs[t], b = uvs[t + 1], c = uvs[t + 2];
        uv_area += 0.5 * std::abs(double((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)));
    }
    return uv_area > 0 ? float(std::sqrt(world_area / uv_area)) : 0.f;
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
    assert(*fh < C.FaceCount);
    const auto vertices = GetVerticesSpan();
    vec3 centroid{0};
    uint32_t count{0};
    for (auto vh : fv_range(fh)) {
        centroid += vertices[*vh].Position;
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

std::vector<float> Mesh::CalcMeanCurvatures(std::span<const uint8_t> edge_sharpness) const {
    std::vector<float> out(VertexCount());
    for (const auto vh : vertices()) out[*vh] = CalcMeanCurvature(vh, edge_sharpness);
    return out;
}

std::optional<double> Mesh::CalcEnclosedVolume() const {
    // A closed manifold surface has exactly two faces per edge.
    uint32_t corners = 0;
    for (const auto fh : faces()) corners += GetValence(fh);
    if (corners == 0 || corners != 2 * EdgeCount()) return std::nullopt;

    // Sum the signed volume of the tetrahedron each triangle spans with the origin. The sign follows the winding.
    double volume = 0;
    ForEachFaceTriangle(*this, [&](uint32_t v0, uint32_t v1, uint32_t v2) {
        const dvec3 a{GetPosition(VH{v0})}, b{GetPosition(VH{v1})}, c{GetPosition(VH{v2})};
        volume += Dot(a, Cross(b, c)) / 6.0;
    });
    return std::abs(volume);
}

he::VH Mesh::FindNearestVertex(vec3 p) const {
    VH closest_vertex;
    float min_dist_sq = std::numeric_limits<float>::max();
    const auto vertex_span = GetVerticesSpan();
    for (const auto vh : vertices()) {
        if (const float dist_sq = Distance2(vertex_span[*vh].Position, p); dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            closest_vertex = vh;
        }
    }
    return closest_vertex;
}

const vec3 &Mesh::GetPosition(VH vh) const { return GetVerticesSpan()[*vh].Position; }
const vec3 &Mesh::GetNormal(VH vh) const { return Store->Arenas().BaseVertexNormals.Get(Store->Get(StoreId).Vertices)[*vh]; }
vec3 Mesh::GetNormal(FH fh) const { return Store->Arenas().BaseFaceNormals.Get(Store->Get(StoreId).FaceData)[*fh]; }
std::span<const Vertex> Mesh::GetVerticesSpan() const { return Store->Arenas().Vertices.Get(Store->Get(StoreId).Vertices); }

VertexAdjacency Mesh::GetVertexEdgeAdjacency() const { return Store->GetVertexEdgeAdjacency(StoreId); }

uint32_t Mesh::TriangleIndexCount() const { return Store->Get(StoreId).TriangleCount * 3; }

void Mesh::WriteTriangleIndices(std::span<uint32_t> dest) const {
    uint32_t i = 0;
    ForEachFaceTriangle(*this, [&](uint32_t v0, uint32_t v1, uint32_t v2) {
        dest[i++] = v0;
        dest[i++] = v1;
        dest[i++] = v2;
    });
}

std::vector<uint32_t> Mesh::CreateTriangleIndices() const {
    uint32_t count = 0;
    for (const auto fh : faces()) count += (GetValence(fh) - 2) * 3;
    std::vector<uint32_t> indices(count);
    WriteTriangleIndices(indices);
    return indices;
}

void Mesh::WriteEdgeIndices(std::span<uint32_t> dest) const {
    uint32_t i = 0;
    for (uint32_t ei = 0; ei < EdgeCount(); ++ei) {
        const auto heh = GetHalfedge(EH{ei}, 0);
        const auto v_from = GetFromVertex(heh);
        const auto v_to = GetToVertex(heh);
        if (!v_from || !v_to) {
            dest[i++] = 0;
            dest[i++] = 0;
            continue;
        }
        dest[i++] = *v_from;
        dest[i++] = *v_to;
    }
}

AABB Mesh::CalcAABB() const {
    AABB b;
    for (const auto &v : GetVerticesSpan()) {
        b.Min = Min(b.Min, v.Position);
        b.Max = Max(b.Max, v.Position);
    }
    return b;
}
