#include "Mesh.h"
#include "MeshComponents.h"
#include "MeshStore.h"

#include <entt/entity/registry.hpp>
#include <glm/gtx/norm.hpp>

#include <algorithm>
#include <bit>

using std::ranges::find_if, std::ranges::distance;

namespace {
constexpr uint64_t MakeEdgeKey(uint32_t from, uint32_t to) { return (uint64_t(from) << 32) | to; }

// Flat open-addressing map reused across connectivity builds via thread_local.
// Key 0 is the empty sentinel (self-loop edges are impossible).
struct EdgeMap {
    void reset(size_t n) {
        const size_t cap = std::bit_ceil(n * 2 | 16u);
        Data.assign(cap, {});
        Mask = cap - 1;
    }
    void insert(uint64_t key, he::HH val) {
        for (auto i = mix(key);; i = (i + 1) & Mask)
            if (!Data[i].Key) {
                Data[i] = {key, val};
                return;
            }
    }
    he::HH *find(uint64_t key) {
        for (auto i = mix(key); Data[i].Key; i = (i + 1) & Mask)
            if (Data[i].Key == key) return &Data[i].Val;
        return nullptr;
    }

private:
    size_t mix(uint64_t k) const {
        // splitmix64 finalizer — avoids clustering from `(from << 32) | to` keys.
        k ^= k >> 30;
        k *= 0xbf58476d1ce4e5b9ULL;
        k ^= k >> 27;
        k *= 0x94d049bb133111ebULL;
        k ^= k >> 31;
        return k & Mask;
    }
    struct Entry {
        uint64_t Key;
        he::HH Val;
    };
    std::vector<Entry> Data;
    size_t Mask = 0;
};
} // namespace

MeshConnectivity BuildConnectivity(std::span<const std::vector<uint32_t>> faces, uint32_t vertex_count) {
    using Halfedge = MeshConnectivity::Halfedge;
    MeshConnectivity c;
    c.VertexCount = vertex_count;
    c.OutgoingHalfedges.resize(vertex_count);

    size_t total_halfedges = 0;
    for (const auto &face : faces) total_halfedges += face.size();
    c.Faces.reserve(faces.size());
    c.Halfedges.reserve(total_halfedges);
    c.HalfedgeToEdge.reserve(total_halfedges);
    c.Edges.reserve(total_halfedges / 2 + 1);
    static thread_local EdgeMap halfedge_map;
    halfedge_map.reset(total_halfedges);
    for (const auto &face : faces) {
        assert(face.size() >= 3);

        const auto fi = c.Faces.size();
        const auto start_he_i = c.Halfedges.size();
        c.Faces.emplace_back(he::HH(start_he_i));

        // Create halfedges, find opposites, and create edges
        for (size_t i = 0; i < face.size(); ++i) {
            const auto to_v = face[i];
            const auto from_v = face[i == 0 ? face.size() - 1 : i - 1];
            c.Halfedges.emplace_back(Halfedge{
                .Vertex = he::VH(to_v),
                .Next = he::HH(start_he_i + (i + 1) % face.size()),
                .Opposite = he::HH{},
                .Face = he::FH(fi),
            });

            const he::HH hh(start_he_i + i);
            if (!c.OutgoingHalfedges[from_v]) c.OutgoingHalfedges[from_v] = hh;
            halfedge_map.insert(MakeEdgeKey(from_v, to_v), hh);

            // Look for opposite halfedge (from previously added faces)
            if (const auto *opposite = halfedge_map.find(MakeEdgeKey(to_v, from_v))) {
                c.Halfedges[*hh].Opposite = *opposite;
                c.Halfedges[**opposite].Opposite = hh;
                c.HalfedgeToEdge.emplace_back(c.HalfedgeToEdge[**opposite]);
            } else {
                // Create new edge
                c.Edges.emplace_back(hh);
                c.HalfedgeToEdge.emplace_back(c.Edges.size() - 1);
            }
        }
    }
    return c;
}

MeshConnectivity BuildConnectivity(std::span<const std::array<uint32_t, 2>> edges, uint32_t vertex_count) {
    using Halfedge = MeshConnectivity::Halfedge;
    MeshConnectivity c;
    c.VertexCount = vertex_count;
    c.OutgoingHalfedges.resize(vertex_count);

    for (const auto &[a, b] : edges) {
        const auto h0 = he::HH(c.Halfedges.size());
        const auto h1 = he::HH(c.Halfedges.size() + 1);
        c.Halfedges.emplace_back(Halfedge{.Vertex = he::VH(b), .Next = he::HH{}, .Opposite = h1, .Face = he::FH{}});
        c.Halfedges.emplace_back(Halfedge{.Vertex = he::VH(a), .Next = he::HH{}, .Opposite = h0, .Face = he::FH{}});

        c.Edges.emplace_back(h0);
        c.HalfedgeToEdge.emplace_back(c.Edges.size() - 1);
        c.HalfedgeToEdge.emplace_back(c.Edges.size() - 1);

        if (!c.OutgoingHalfedges[a]) c.OutgoingHalfedges[a] = h0;
        if (!c.OutgoingHalfedges[b]) c.OutgoingHalfedges[b] = h1;
    }
    return c;
}

Mesh GetMesh(const entt::registry &r, entt::entity e) {
    return {r.ctx().get<const MeshStore>(), r.get<const MeshHandle>(e).StoreId, r.get<const MeshConnectivity>(e)};
}
std::optional<Mesh> TryGetMesh(const entt::registry &r, entt::entity e) {
    const auto *handle = r.try_get<const MeshHandle>(e);
    if (!handle) return std::nullopt;
    return Mesh{r.ctx().get<const MeshStore>(), handle->StoreId, r.get<const MeshConnectivity>(e)};
}
bool HasMesh(const entt::registry &r, entt::entity e) { return r.all_of<MeshHandle>(e); }

he::VH Mesh::GetFromVertex(HH hh) const {
    assert(*hh < C->Halfedges.size());
    if (const auto opp = C->Halfedges[*hh].Opposite) return C->Halfedges[*opp].Vertex;

    // For boundary halfedges, find the previous halfedge in the face loop
    const auto range = FaceHalfedgeRange{this, C->Halfedges[*hh].Next};
    auto it = find_if(range, [&](HH h) { return C->Halfedges[*h].Next == hh; });
    return it != range.end() ? C->Halfedges[**it].Vertex : VH{};
}

uint32_t Mesh::GetValence(VH vh) const { return distance(voh_range(vh)); }
uint32_t Mesh::GetValence(FH fh) const { return distance(fh_range(fh)); }

vec3 Mesh::CalcFaceCentroid(FH fh) const {
    assert(*fh < C->Faces.size());
    const auto vertices = Store->GetVertices(StoreId);
    vec3 centroid{0};
    uint32_t count{0};
    for (auto vh : fv_range(fh)) {
        centroid += vertices[*vh].Position;
        count++;
    }
    return count > 0 ? centroid / float(count) : centroid;
}

float Mesh::CalcEdgeLength(HH hh) const {
    assert(*hh < C->Halfedges.size());
    const auto vertices = GetVerticesSpan();
    const auto from_v = GetFromVertex(hh);
    const auto to_v = C->Halfedges[*hh].Vertex;
    if (!from_v || !to_v) return 0;
    return glm::length(vertices[*to_v].Position - vertices[*from_v].Position);
}

float Mesh::CalcFaceArea(FH fh) const {
    assert(*fh < FaceCount());
    const auto vertices = GetVerticesSpan();
    float area{0};
    auto fv_it = cfv_iter(fh);
    const auto p0 = vertices[**fv_it++].Position;
    for (vec3 p1 = vertices[**fv_it++].Position, p2; fv_it; ++fv_it) {
        p2 = vertices[**fv_it].Position;
        area += glm::length(glm::cross(p1 - p0, p2 - p0)) * 0.5f;
        p1 = p2;
    }
    return area;
}

he::VH Mesh::FindNearestVertex(vec3 p) const {
    VH closest_vertex;
    float min_dist_sq = std::numeric_limits<float>::max();
    const auto vertex_span = GetVerticesSpan();
    for (const auto vh : vertices()) {
        if (const float dist_sq = glm::distance2(vertex_span[*vh].Position, p); dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            closest_vertex = vh;
        }
    }
    return closest_vertex;
}

const vec3 &Mesh::GetPosition(VH vh) const { return GetVerticesSpan()[*vh].Position; }
const vec3 &Mesh::GetNormal(VH vh) const { return GetVerticesSpan()[*vh].Normal; }
vec3 Mesh::GetNormal(FH fh) const {
    auto it = cfv_iter(fh);
    const auto p0 = GetPosition(*it), p1 = GetPosition(*++it), p2 = GetPosition(*++it);
    return glm::normalize(glm::cross(p1 - p0, p2 - p0));
}
std::span<const Vertex> Mesh::GetVerticesSpan() const { return Store->GetVertices(StoreId); }

uint32_t Mesh::TriangleIndexCount() const { return Store->GetTriangleCount(StoreId) * 3; }

void Mesh::WriteTriangleIndices(std::span<uint32_t> dest) const {
    uint32_t i = 0;
    for (const auto fh : faces()) {
        auto fv_it = cfv_iter(fh);
        const auto v0 = *fv_it++;
        VH v1 = *fv_it++, v2;
        for (; fv_it; ++fv_it) {
            v2 = *fv_it;
            dest[i++] = *v0;
            dest[i++] = *v1;
            dest[i++] = *v2;
            v1 = v2;
        }
    }
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

BBox Mesh::GetBBox() const {
    BBox b;
    for (const auto &v : GetVerticesSpan()) {
        b.Min = glm::min(b.Min, v.Position);
        b.Max = glm::max(b.Max, v.Position);
    }
    return b;
}
