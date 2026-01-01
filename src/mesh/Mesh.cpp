#include "Mesh.h"
#include "MeshStore.h"

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include <algorithm>
#include <cassert>
#include <ranges>
#include <utility>

using std::ranges::any_of, std::ranges::find_if, std::ranges::distance;

namespace {
constexpr uint64_t MakeEdgeKey(uint from, uint to) {
    return (static_cast<uint64_t>(from) << 32) | static_cast<uint64_t>(to);
}
} // namespace

Mesh::Mesh(MeshStore &store, uint32_t store_id, std::vector<std::vector<uint>> &&faces)
    : Store(&store), StoreId(store_id),
      Vertices(store.GetVertices(store_id)),
      FaceNormals(store.GetFaceNormals(store_id)) {
    OutgoingHalfedges.resize(Vertices.size());

    std::unordered_map<uint64_t, HH> halfedge_map;
    for (const auto &face : faces) {
        assert(face.size() >= 3);

        const auto fi = Faces.size();
        const auto start_he_i = Halfedges.size();
        Faces.emplace_back(HH(start_he_i));

        // Create halfedges, find opposites, and create edges
        for (size_t i = 0; i < face.size(); ++i) {
            const auto to_v = face[i];
            const auto from_v = face[i == 0 ? face.size() - 1 : i - 1];
            Halfedges.emplace_back(Halfedge{
                .Vertex = VH(to_v),
                .Next = HH(start_he_i + (i + 1) % face.size()),
                .Opposite = HH{},
                .Face = FH(fi),
            });

            const HH hh(start_he_i + i);
            if (!OutgoingHalfedges[from_v]) OutgoingHalfedges[from_v] = hh;
            halfedge_map.emplace(MakeEdgeKey(from_v, to_v), hh);

            // Look for opposite halfedge (from previously added faces)
            if (const auto it = halfedge_map.find(MakeEdgeKey(to_v, from_v));
                it != halfedge_map.end()) {
                // Found existing opposite halfedge
                const auto opposite_hh = it->second;
                Halfedges[*hh].Opposite = opposite_hh;
                Halfedges[*opposite_hh].Opposite = hh;
                // They should share the same edge
                HalfedgeToEdge.emplace_back(HalfedgeToEdge[*opposite_hh]);
            } else {
                // Create new edge
                Edges.emplace_back(hh);
                HalfedgeToEdge.emplace_back(Edges.size() - 1);
            }
        }
    }
}

Mesh::Mesh(MeshStore &store, uint32_t store_id, const Mesh &src)
    : Store(&store), StoreId(store_id),
      Vertices(store.GetVertices(store_id)),
      FaceNormals(store.GetFaceNormals(store_id)),
      OutgoingHalfedges(src.OutgoingHalfedges),
      Halfedges(src.Halfedges),
      HalfedgeToEdge(src.HalfedgeToEdge),
      Edges(src.Edges),
      Faces(src.Faces),
      Color(src.Color) {}

Mesh::Mesh(Mesh &&other) noexcept
    : Store(std::exchange(other.Store, nullptr)),
      StoreId(std::exchange(other.StoreId, InvalidStoreId)),
      Vertices(std::exchange(other.Vertices, {})),
      FaceNormals(std::exchange(other.FaceNormals, {})),
      OutgoingHalfedges(std::move(other.OutgoingHalfedges)),
      Halfedges(std::move(other.Halfedges)),
      HalfedgeToEdge(std::move(other.HalfedgeToEdge)),
      Edges(std::move(other.Edges)),
      Faces(std::move(other.Faces)),
      Color(other.Color) {}

Mesh &Mesh::operator=(Mesh &&other) noexcept {
    if (this != &other) {
        if (Store && StoreId != InvalidStoreId) Store->Release(StoreId);
        Store = std::exchange(other.Store, nullptr);
        StoreId = std::exchange(other.StoreId, InvalidStoreId);
        Vertices = std::exchange(other.Vertices, {});
        FaceNormals = std::exchange(other.FaceNormals, {});
        OutgoingHalfedges = std::move(other.OutgoingHalfedges);
        Halfedges = std::move(other.Halfedges);
        HalfedgeToEdge = std::move(other.HalfedgeToEdge);
        Edges = std::move(other.Edges);
        Faces = std::move(other.Faces);
        Color = other.Color;
    }
    return *this;
}

Mesh::~Mesh() {
    if (Store && StoreId != InvalidStoreId) Store->Release(StoreId);
}

he::VH Mesh::GetFromVertex(HH hh) const {
    assert(*hh < Halfedges.size());
    const auto opp = Halfedges[*hh].Opposite;
    if (opp) return Halfedges[*opp].Vertex;

    // For boundary halfedges, find the previous halfedge in the face loop
    const auto range = FaceHalfedgeRange{this, Halfedges[*hh].Next};
    auto it = find_if(range, [&](HH h) { return Halfedges[*h].Next == hh; });
    return it != range.end() ? Halfedges[**it].Vertex : VH{};
}

uint Mesh::GetValence(VH vh) const { return distance(voh_range(vh)); }
uint Mesh::GetValence(FH fh) const { return distance(fh_range(fh)); }

vec3 Mesh::CalcFaceCentroid(FH fh) const {
    assert(*fh < Faces.size());
    vec3 centroid{0};
    uint count{0};
    for (auto vh : fv_range(fh)) {
        centroid += Vertices[*vh].Position;
        count++;
    }
    return count > 0 ? centroid / static_cast<float>(count) : centroid;
}

float Mesh::CalcEdgeLength(HH hh) const {
    assert(*hh < Halfedges.size());
    const auto from_v = GetFromVertex(hh);
    const auto to_v = Halfedges[*hh].Vertex;
    if (!from_v || !to_v) return 0;
    return glm::length(Vertices[*to_v].Position - Vertices[*from_v].Position);
}

float Mesh::CalcFaceArea(FH fh) const {
    assert(*fh < FaceCount());
    float area{0};
    auto fv_it = cfv_iter(fh);
    const auto p0 = Vertices[**fv_it++].Position;
    for (vec3 p1 = Vertices[**fv_it++].Position, p2; fv_it; ++fv_it) {
        p2 = Vertices[**fv_it].Position;
        area += glm::length(glm::cross(p1 - p0, p2 - p0)) * 0.5f;
        p1 = p2;
    }
    return area;
}

he::VH Mesh::FindNearestVertex(vec3 p) const {
    VH closest_vertex;
    float min_distance_sq = std::numeric_limits<float>::max();
    for (const auto vh : vertices()) {
        const vec3 diff = Vertices[*vh].Position - p;
        if (const float distance_sq = glm::dot(diff, diff); distance_sq < min_distance_sq) {
            min_distance_sq = distance_sq;
            closest_vertex = vh;
        }
    }
    return closest_vertex;
}

bool Mesh::VertexBelongsToFace(VH vh, FH fh) const {
    return vh && fh && any_of(fv_range(fh), [vh](const auto &fv) { return fv == vh; });
}

bool Mesh::VertexBelongsToEdge(VH vh, EH eh) const {
    return vh && eh && any_of(voh_range(vh), [&](const auto &heh) { return GetEdge(heh) == eh; });
}

bool Mesh::VertexBelongsToFaceEdge(VH vh, FH fh, EH eh) const {
    return fh && eh && any_of(voh_range(vh), [&](const auto &heh) {
               return GetEdge(heh) == eh && (GetFace(heh) == fh || GetFace(GetOppositeHalfedge(heh)) == fh);
           });
}

bool Mesh::EdgeBelongsToFace(EH eh, FH fh) const {
    return eh && fh && any_of(fh_range(fh), [&](const auto &heh) { return GetEdge(heh) == eh; });
}

std::vector<uint> Mesh::CreateTriangleIndices() const {
    std::vector<uint> indices;
    for (const auto fh : faces()) {
        auto fv_it = cfv_iter(fh);
        const auto v0 = *fv_it++;
        VH v1 = *fv_it++, v2;
        for (; fv_it; ++fv_it) {
            v2 = *fv_it;
            indices.insert(indices.end(), {*v0, *v1, *v2});
            v1 = v2;
        }
    }
    return indices;
}

std::vector<uint> Mesh::CreateTriangulatedFaceIndices() const {
    std::vector<uint> indices;
    uint index = 0;
    for (const auto fh : faces()) {
        const auto valence = GetValence(fh);
        for (uint i = 0; i < valence - 2; ++i) {
            indices.insert(indices.end(), {index, index + i + 1, index + i + 2});
        }
        index += valence;
    }
    return indices;
}

std::vector<uint> Mesh::CreateEdgeIndices() const {
    std::vector<uint> indices;
    indices.reserve(EdgeCount() * 2);
    for (uint ei = 0; ei < EdgeCount(); ++ei) {
        const auto heh = GetHalfedge(EH{ei}, 0);
        const auto v_from = GetFromVertex(heh);
        const auto v_to = GetToVertex(heh);
        if (!v_from || !v_to) {
            indices.emplace_back(0);
            indices.emplace_back(0);
            continue;
        }
        indices.emplace_back(*v_from);
        indices.emplace_back(*v_to);
    }
    return indices;
}
