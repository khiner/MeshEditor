#include "Mesh.h"

#include "MeshComponents.h"
#include "MeshStore.h"
#include "Parallel.h"

#include <entt/entity/registry.hpp>
#include <glm/gtx/norm.hpp>

#include <algorithm>

using std::ranges::distance;

namespace {
// The edge endpoint that is not the bucket's vertex, with the top bit set when the halfedge runs
// from the higher endpoint to the lower one. Two halfedges of one edge share the endpoint and differ
// in the bit, so a bucket scan pairs them.
constexpr uint32_t ReverseBit{1u << 31};

// Mark each edge's first halfedge and count the marks before every word, so an edge index is a rank.
// Both builders append to Edges in ascending first-halfedge order, which is what makes the count run.
void BuildEdgeRanks(MeshConnectivity &c) {
    const auto words = uint32_t((c.Opposites.size() + 31u) / 32u);
    c.EdgeFirstBits.assign(words, 0u);
    c.EdgeFirstRanks.assign(words, 0u);
    uint32_t edge = 0;
    c.EdgeSamples.assign((c.Edges.size() + 31u) / 32u, 0u);
    for (uint32_t word = 0; word < words; ++word) {
        c.EdgeFirstRanks[word] = edge;
        uint32_t bits = 0;
        while (edge < c.Edges.size() && *c.Edges[edge] / 32u == word) {
            if (edge % 32u == 0) c.EdgeSamples[edge / 32u] = word;
            bits |= 1u << (*c.Edges[edge++] % 32u);
        }
        c.EdgeFirstBits[word] = bits;
    }
    c.EdgeCount = uint32_t(c.Edges.size());
}

} // namespace

MeshConnectivity BuildConnectivity(std::span<const uint32_t> face_offsets, std::span<const uint32_t> face_corners, uint32_t vertex_count) {
    assert(vertex_count < ReverseBit);
    const auto face_count = uint32_t(face_offsets.size() - 1);
    MeshConnectivity c;
    c.VertexCount = vertex_count;
    c.OutgoingHalfedges.resize(vertex_count);

    // One halfedge per face corner, in corner order, so a corner's index is its halfedge's index.
    const auto total_halfedges = face_corners.size();
    c.Opposites.assign(total_halfedges, he::HH{});

    const auto for_each_halfedge = [&](auto &&body) {
        for (uint32_t f = 0; f < face_count; ++f) {
            const auto first = face_offsets[f], last = face_offsets[f + 1];
            for (auto h = first; h < last; ++h) body(h, face_corners[h == first ? last - 1 : h - 1], face_corners[h]);
        }
    };

    // A triangle mesh's face f always starts at halfedge 3f, so its face table is arithmetic.
    const bool all_triangles = total_halfedges == 3 * size_t(face_count);
    c.FaceCount = face_count;
    if (!all_triangles) {
        c.Faces.reserve(face_count);
        for (uint32_t f = 0; f < face_count; ++f) {
            assert(face_offsets[f + 1] - face_offsets[f] >= 3);
            c.Faces.emplace_back(he::HH(face_offsets[f]));
        }
    }
    for_each_halfedge([&](uint32_t h, uint32_t from_v, uint32_t) {
        if (!c.OutgoingHalfedges[from_v]) c.OutgoingHalfedges[from_v] = he::HH(h);
    });

    // Group halfedges by their lower endpoint with a counting sort. Both halfedges of an edge land in
    // the same bucket, in ascending halfedge order, so one sequential scan of a bucket finds every pair.
    const auto halfedge_count = uint32_t(total_halfedges);
    std::vector<uint32_t> bucket_offsets(size_t(vertex_count) + 1, 0u);
    for_each_halfedge([&](uint32_t, uint32_t from_v, uint32_t to_v) { ++bucket_offsets[std::min(from_v, to_v) + 1]; });
    for (uint32_t v = 0; v < vertex_count; ++v) bucket_offsets[v + 1] += bucket_offsets[v];
    std::vector<uint32_t> bucket_halfedges(halfedge_count);
    {
        std::vector<uint32_t> cursor(bucket_offsets.begin(), bucket_offsets.end() - 1);
        for_each_halfedge([&](uint32_t h, uint32_t from_v, uint32_t to_v) { bucket_halfedges[cursor[std::min(from_v, to_v)]++] = h; });
    }

    // A halfedge's endpoints come back from its own index, so a bucket holds only the index.
    const auto bucket_key = [&](uint32_t h) {
        const auto f = all_triangles ? h / 3u : uint32_t(std::upper_bound(face_offsets.begin(), face_offsets.end(), h) - face_offsets.begin()) - 1u;
        const auto first = face_offsets[f], last = face_offsets[f + 1];
        const auto to_v = face_corners[h], from_v = face_corners[h == first ? last - 1u : h - 1u];
        return std::max(from_v, to_v) | (from_v > to_v ? ReverseBit : 0u);
    };

    // A halfedge joins the edge of the first halfedge that runs the other way, and starts one of its
    // own when there is none before it. On a manifold edge, carrying exactly two halfedges, that
    // first one is the lower of the pair, so only a wider bucket needs its edges written down.
    constexpr uint32_t VertexBlock{4u * 1024u};
    const auto block_count = (vertex_count + VertexBlock - 1u) / VertexBlock;
    std::vector<uint8_t> block_shares_an_edge(block_count, 0u);
    const auto pair_bucket = [&](uint32_t v, std::vector<uint32_t> &keys, auto &&joined) {
        const auto first = bucket_offsets[v], last = bucket_offsets[v + 1];
        keys.clear();
        for (auto p = first; p < last; ++p) keys.emplace_back(bucket_key(bucket_halfedges[p]));
        for (uint32_t i = 0; i < keys.size(); ++i) {
            const auto key = keys[i];
            uint32_t opposite = he::null, sharing = 0;
            for (uint32_t j = 0; j < keys.size(); ++j) {
                if (keys[j] == (key ^ ReverseBit) && opposite == he::null) opposite = bucket_halfedges[first + j];
                sharing += (keys[j] & ~ReverseBit) == (key & ~ReverseBit);
            }
            joined(bucket_halfedges[first + i], opposite, sharing > 2u);
        }
    };
    ParallelFor(block_count, [&](uint32_t block) {
        const auto block_last = std::min((block + 1u) * VertexBlock, vertex_count);
        std::vector<uint32_t> keys;
        for (auto v = block * VertexBlock; v < block_last; ++v) {
            pair_bucket(v, keys, [&](uint32_t h, uint32_t opposite, bool shared) {
                if (shared) block_shares_an_edge[block] = 1u;
                if (opposite < h) {
                    c.Opposites[h] = he::HH(opposite);
                    c.Opposites[opposite] = he::HH(h);
                }
            });
        }
    });

    // Edges number by ascending first halfedge, which is the order the halfedge walk reaches them.
    c.Edges.reserve(halfedge_count / 2 + 1);
    const bool ranks_answer = std::ranges::none_of(block_shares_an_edge, [](uint8_t f) { return f != 0u; });
    std::vector<uint32_t> edge_representative;
    if (ranks_answer) {
        for (uint32_t h = 0; h < halfedge_count; ++h) {
            if (const auto opposite = c.Opposites[h]; !opposite || *opposite > h) c.Edges.emplace_back(he::HH(h));
        }
    } else {
        // A shared edge leaves a halfedge whose first is neither itself nor its opposite, so walk the
        // buckets again and record where each one landed.
        edge_representative.assign(halfedge_count, 0u);
        std::vector<uint32_t> keys;
        for (uint32_t v = 0; v < vertex_count; ++v) {
            pair_bucket(v, keys, [&](uint32_t h, uint32_t opposite, bool) {
                edge_representative[h] = opposite < h ? edge_representative[opposite] : h;
            });
        }
        for (uint32_t h = 0; h < halfedge_count; ++h) {
            if (edge_representative[h] == h) c.Edges.emplace_back(he::HH(h));
        }
    }
    BuildEdgeRanks(c);
    if (ranks_answer) c.Edges = std::vector<he::HH>{};
    else {
        c.EdgeFirstBits = std::vector<uint32_t>{};
        c.EdgeFirstRanks = std::vector<uint32_t>{};
        c.HalfedgeToEdge.resize(halfedge_count);
        uint32_t edge = 0;
        for (uint32_t h = 0; h < halfedge_count; ++h) {
            c.HalfedgeToEdge[h] = edge_representative[h] == h ? he::EH(edge++) : c.HalfedgeToEdge[edge_representative[h]];
        }
    }
    return c;
}

MeshConnectivity BuildConnectivity(std::span<const std::array<uint32_t, 2>> edges, uint32_t vertex_count) {
    MeshConnectivity c;
    c.VertexCount = vertex_count;
    c.OutgoingHalfedges.resize(vertex_count);

    for (const auto &[a, b] : edges) {
        const auto h0 = he::HH(c.Opposites.size());
        const auto h1 = he::HH(c.Opposites.size() + 1);
        c.Opposites.emplace_back(h1);
        c.Opposites.emplace_back(h0);

        c.Edges.emplace_back(h0);

        if (!c.OutgoingHalfedges[a]) c.OutgoingHalfedges[a] = h0;
        if (!c.OutgoingHalfedges[b]) c.OutgoingHalfedges[b] = h1;
    }
    BuildEdgeRanks(c);
    c.Edges = std::vector<he::HH>{};
    return c;
}

Mesh::Mesh(const MeshStore &store, uint32_t store_id, const MeshConnectivity &c)
    : Store(&store), StoreId(store_id), C(&c), Corners(store.GetFaceCorners(store_id)) {}

Mesh GetMesh(const entt::registry &r, entt::entity e) {
    return {r.ctx().get<const MeshStore>(), r.get<const MeshHandle>(e).StoreId, r.get<const MeshConnectivity>(e)};
}
std::optional<Mesh> TryGetMesh(const entt::registry &r, entt::entity e) {
    const auto *handle = r.try_get<const MeshHandle>(e);
    if (!handle) return std::nullopt;
    return Mesh{r.ctx().get<const MeshStore>(), handle->StoreId, r.get<const MeshConnectivity>(e)};
}
bool HasMesh(const entt::registry &r, entt::entity e) { return r.all_of<MeshHandle>(e); }

float LocalLengthPerUv(const entt::registry &r, entt::entity mesh_entity, uint32_t uv_set) {
    const auto mesh = TryGetMesh(r, mesh_entity);
    if (!mesh || uv_set >= MeshStore::MaxUvSets) return 0;
    const auto uvs = r.ctx().get<const MeshStore>().GetCornerUvs(mesh->GetStoreId(), uv_set);
    const auto corners = mesh->CreateTriangleIndices();
    if (uvs.size() != corners.size() || corners.empty()) return 0;

    double world_area = 0, uv_area = 0;
    for (size_t t = 0; t + 2 < corners.size(); t += 3) {
        const vec3 p0 = mesh->GetPosition(Mesh::VH{corners[t]});
        const vec3 p1 = mesh->GetPosition(Mesh::VH{corners[t + 1]});
        const vec3 p2 = mesh->GetPosition(Mesh::VH{corners[t + 2]});
        world_area += 0.5 * double(glm::length(glm::cross(p1 - p0, p2 - p0)));
        const vec2 a = uvs[t], b = uvs[t + 1], c = uvs[t + 2];
        uv_area += 0.5 * std::abs(double((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)));
    }
    return uv_area > 0 ? float(std::sqrt(world_area / uv_area)) : 0.f;
}

he::VH Mesh::GetFromVertex(HH hh) const {
    assert(*hh < C->Opposites.size());
    if (const auto opp = C->Opposites[*hh]) return VH(Corners[*opp]);
    // A boundary halfedge takes it from the halfedge before it in the face loop.
    const auto prev = C->Previous(hh);
    return prev ? VH(Corners[*prev]) : VH{};
}

uint32_t Mesh::GetValence(VH vh) const { return distance(voh_range(vh)); }
uint32_t Mesh::GetValence(FH fh) const { return distance(fh_range(fh)); }

vec3 Mesh::CalcFaceCentroid(FH fh) const {
    assert(*fh < C->FaceCount);
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
    assert(*hh < C->Opposites.size());
    const auto vertices = GetVerticesSpan();
    const auto from_v = GetFromVertex(hh);
    const auto to_v = VH(Corners[*hh]);
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

float Mesh::CalcMeanCurvature(VH vh, std::span<const uint8_t> edge_sharpness) const {
    // A sharp edge is a shading discontinuity, so a vertex on one sits on a crease between faces rather than inside a smooth surface.
    // Curvature belongs to the smooth surface, and the faces meeting at a crease are each flat where they reach it.
    for (const auto he : voh_range(vh)) {
        const auto eh = GetEdge(he);
        if (*eh < edge_sharpness.size() && edge_sharpness[*eh] != 0) return 0.f;
    }

    const vec3 xi = GetPosition(vh);
    const vec3 ni = glm::normalize(GetNormal(vh));
    double sum = 0;
    int count = 0;
    for (const auto he : voh_range(vh)) {
        // A halfedge with no opposite bounds the surface rather than running through it.
        if (!GetOppositeHalfedge(he)) continue;
        const vec3 d = GetPosition(GetToVertex(he)) - xi;
        const double d2 = glm::dot(d, d);
        if (d2 < 1e-20) continue;
        sum += -2.0 * double(glm::dot(d, ni)) / d2;
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
    // A closed manifold surface has exactly two faces per edge. Open and non-manifold surfaces both fail this.
    uint32_t corners = 0;
    for (const auto fh : faces()) corners += GetValence(fh);
    if (corners == 0 || corners != 2 * EdgeCount()) return std::nullopt;

    // Sum the signed volume of the tetrahedron each triangle spans with the origin. The sign follows the winding.
    double volume = 0;
    for (const auto fh : faces()) {
        auto fv_it = cfv_iter(fh);
        const auto v0 = *fv_it++;
        VH v1 = *fv_it++, v2;
        for (; fv_it; ++fv_it) {
            v2 = *fv_it;
            const glm::dvec3 a{GetPosition(v0)}, b{GetPosition(v1)}, c{GetPosition(v2)};
            volume += glm::dot(a, glm::cross(b, c)) / 6.0;
            v1 = v2;
        }
    }
    return std::abs(volume);
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
const vec3 &Mesh::GetNormal(VH vh) const { return Store->GetBaseVertexNormals(StoreId)[*vh]; }
vec3 Mesh::GetNormal(FH fh) const { return Store->GetBaseFaceNormals(StoreId)[*fh]; }
std::span<const Vertex> Mesh::GetVerticesSpan() const { return Store->GetVertices(StoreId); }

VertexAdjacency Mesh::GetVertexEdgeAdjacency() const { return Store->GetVertexEdgeAdjacency(StoreId); }

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

AABB Mesh::CalcAABB() const {
    AABB b;
    for (const auto &v : GetVerticesSpan()) {
        b.Min = glm::min(b.Min, v.Position);
        b.Max = glm::max(b.Max, v.Position);
    }
    return b;
}
