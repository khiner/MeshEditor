#include "render/ClusterLod.h"
#include "numeric/VectorMath.h"

#include "FlatKeyMap.h"
#include "Parallel.h"
#include "Profile.h"
#include "gpu/MeshAttributeBit.h"
#include "gpu/MeshletGeometryEncoding.h"

#include "meshoptimizer.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <exception>
#include <format>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace {
constexpr size_t PartitionSize{ClusterLodPartitionSize};
constexpr float SimplifyRatio{0.5f};
constexpr float SimplifyThreshold{0.85f};
constexpr float ClusterConeWeight{0.5f};
constexpr float ClusterSplitFactor{2.f};
constexpr float ShadingAttributeScale{16.f};
constexpr float SpanNodeSlack{1.f + 1e-5f};
// Use nested parallelism only when one weld can occupy the machine independently.
constexpr uint32_t ParallelPositionRemapVertices{256u * 1024u};

uint32_t PositionHash(const std::array<float, 3> &position) {
    std::array<uint32_t, 3> bits;
    for (uint32_t i = 0; i < bits.size(); ++i) {
        bits[i] = std::bit_cast<uint32_t>(position[i]);
        if (bits[i] == 0x80000000u) bits[i] = 0u;
        bits[i] ^= bits[i] >> 17u;
    }
    return (bits[0] * 73856093u) ^ (bits[1] * 19349663u) ^ (bits[2] * 83492791u);
}

bool SamePosition(const std::array<float, 3> &a, const std::array<float, 3> &b) {
    return a == b;
}

void GeneratePositionRemap(std::vector<uint32_t> &remap, const std::vector<std::array<float, 3>> &positions) {
    constexpr uint32_t BlockSize{16u * 1024u};
    const uint32_t count = uint32_t(positions.size());
    const size_t table_size = std::bit_ceil(size_t(count) + count / 4u);
    const size_t mask = table_size - 1u;
    std::vector<uint32_t> table(table_size, ClusterLodInvalid);
    const uint32_t block_count = (count + BlockSize - 1u) / BlockSize;
    const auto for_each_position = [&](auto &&function) {
        ParallelFor(block_count, [&](uint32_t block) {
            const uint32_t last = std::min((block + 1u) * BlockSize, count);
            for (uint32_t i = block * BlockSize; i < last; ++i) function(i);
        });
    };
    // Equal positions follow the same probe sequence and atomically select the lowest input index.
    for_each_position([&](uint32_t i) {
        size_t bucket = PositionHash(positions[i]) & mask;
        for (size_t probe = 0; probe <= mask; ++probe) {
            std::atomic_ref entry{table[bucket]};
            uint32_t occupant = entry.load(std::memory_order_relaxed);
            if (occupant == ClusterLodInvalid) {
                if (entry.compare_exchange_strong(occupant, i, std::memory_order_relaxed)) break;
            }
            if (occupant == i || SamePosition(positions[occupant], positions[i])) {
                while (occupant > i && !entry.compare_exchange_weak(occupant, i, std::memory_order_relaxed)) {}
                break;
            }
            bucket = (bucket + probe + 1u) & mask;
        }
    });
    for_each_position([&](uint32_t i) {
        size_t bucket = PositionHash(positions[i]) & mask;
        for (size_t probe = 0; probe <= mask; ++probe) {
            const uint32_t occupant = table[bucket];
            assert(occupant != ClusterLodInvalid);
            if (occupant == i || SamePosition(positions[occupant], positions[i])) {
                remap[i] = occupant;
                break;
            }
            bucket = (bucket + probe + 1u) & mask;
        }
    });
}

// Preserves meshopt's center-then-radius sphere layout and adds simplification error.
struct Bounds {
    vec3 Center{};
    float Radius{};
    float Error{};
};
static_assert(offsetof(Bounds, Radius) == sizeof(vec3));

Bounds MergeBounds(std::span<const Bounds> bounds) {
    const auto merged = meshopt_computeSphereBounds(&bounds.front().Center.x, bounds.size(), sizeof(Bounds), &bounds.front().Radius, sizeof(Bounds));
    Bounds result{.Center = std::bit_cast<vec3>(merged.center), .Radius = merged.radius};
    // Merged bounds stay conservative with respect to the member errors.
    for (const auto &member : bounds) result.Error = std::max(result.Error, member.Error);
    return result;
}

// Produces bounds that contain every member sphere and error.
Bounds MergeSpanBounds(std::span<const Bounds> members) {
    const auto merged = meshopt_computeSphereBounds(&members.front().Center.x, members.size(), sizeof(Bounds), &members.front().Radius, sizeof(Bounds));
    Bounds result{.Center = std::bit_cast<vec3>(merged.center)};
    for (const auto &member : members) {
        const float dx = member.Center[0] - result.Center[0];
        const float dy = member.Center[1] - result.Center[1];
        const float dz = member.Center[2] - result.Center[2];
        result.Radius = std::max(result.Radius, std::sqrt(dx * dx + dy * dy + dz * dz) + member.Radius);
        result.Error = std::max(result.Error, member.Error);
    }
    // Add rounding slack so a node cannot prune a record that exceeds the projected-error budget.
    result.Radius *= SpanNodeSlack;
    result.Error *= SpanNodeSlack;
    return result;
}

// The whole-primitive render-vertex weld, which is the domain every simplification runs in.
struct PrimitiveWeld {
    std::vector<uint32_t> CornerVertices; // weld vertex per primitive corner
    std::vector<uint32_t> Representative; // primitive-local source corner per weld vertex
    std::vector<std::array<float, 3>> Positions;
    std::vector<float> Attributes;
    uint32_t AttributeCount{};
    std::vector<uint32_t> Remap; // canonical weld vertex sharing a position
    std::vector<uint8_t> Locks;
    std::vector<uint8_t> SeamLocks;
    std::vector<uint32_t> Owners; // the first pending group to use each canonical weld vertex, while locks derive
    std::vector<uint32_t> SourceVertices;
    uint32_t VertexFirst{};
    FlatKeyMap Keys;
    TriangleVertexView Source;
    uint32_t FirstCorner{};
    uint32_t CanonicalCorner(uint32_t corner) const {
        return Source.Corners.Values.empty() ? FirstCorner + corner : Source.Corners[FirstCorner + corner];
    }
    float Scale{}; // Extent factor from meshopt_simplifyScale, converting normalized weights to mesh units.

    uint32_t VertexCount() const { return uint32_t(Representative.size()); }
};

// Uses the shared render-equivalence key for both source and coarse cluster vertices.
void BuildWeld(const ClusterLodMesh &mesh, const ClusterLodPrimitive &primitive, PrimitiveWeld &weld, bool serial) {
    const uint32_t first_index = primitive.FirstTriangle * 3u;
    const uint32_t corner_count = primitive.TriangleCount * 3u;
    const auto primitive_indices = mesh.CornerVertices.subspan(first_index, corner_count);
    const CornerWeldKey key{mesh.Weld, first_index};

    weld.Source = mesh.CornerVertices;
    weld.FirstCorner = first_index;
    weld.SourceVertices.clear();
    weld.CornerVertices.assign(corner_count, 0u);
    weld.Representative.clear();
    weld.Positions.clear();
    const auto append_render_vertex = [&](uint32_t corner, uint32_t source_vertex) {
        const uint32_t render_vertex = uint32_t(weld.Representative.size());
        weld.CornerVertices[corner] = render_vertex;
        weld.Representative.push_back(corner);
        const float *position = mesh.Positions + (mesh.PositionStride / sizeof(float)) * (source_vertex - mesh.VertexFirst);
        weld.Positions.push_back({position[0], position[1], position[2]});
        return render_vertex;
    };
    uint64_t vertex_span = mesh.DenseVertices.Count;
    weld.VertexFirst = mesh.DenseVertices.Offset;
    if (!vertex_span) {
        // Blocks bound their corners' source vertices independently, and the bounds combine in block order.
        constexpr uint32_t BoundCorners{64u * 1024u};
        const uint32_t block_count = (corner_count + BoundCorners - 1u) / BoundCorners;
        std::vector<std::pair<uint32_t, uint32_t>> bounds(block_count, {InvalidOffset, 0u});
        const auto bound = [&](uint32_t block) {
            auto &[first, last] = bounds[block];
            for (uint32_t c = block * BoundCorners; c < std::min(corner_count, (block + 1u) * BoundCorners); ++c) {
                first = std::min(first, primitive_indices[c]);
                last = std::max(last, primitive_indices[c]);
            }
        };
        if (serial) {
            for (uint32_t block = 0; block < block_count; ++block) bound(block);
        } else {
            ParallelFor(block_count, bound);
        }
        uint32_t max_source_vertex = 0;
        weld.VertexFirst = InvalidOffset;
        for (const auto &[first, last] : bounds) {
            weld.VertexFirst = std::min(weld.VertexFirst, first);
            max_source_vertex = std::max(max_source_vertex, last);
        }
        vertex_span = uint64_t(max_source_vertex) - weld.VertexFirst + 1u;
    }
    const bool dense_source_vertices = vertex_span <= size_t(corner_count) * 2u;
    const bool uniform_face = mesh.Weld.CornerClassMode == uint32_t(CornerClassMode::UniformFace);
    const bool source_vertex_only = key.WordCount() == 3u && mesh.Weld.CornerClassMode != uint32_t(CornerClassMode::Mixed) &&
        mesh.Weld.CustomNormals.empty() && (!uniform_face || !mesh.Weld.MorphShadingAuthored);
    if (source_vertex_only && dense_source_vertices) {
        weld.SourceVertices.assign(size_t(vertex_span), ClusterLodInvalid);
        if (!serial && corner_count >= 256u * 1024u) {
            constexpr uint32_t BlockTriangles{16u * 1024u};
            const uint32_t block_count = (primitive.TriangleCount + BlockTriangles - 1u) / BlockTriangles;
            const auto for_each_triangle_block = [&](auto &&body) {
                ParallelFor(block_count, [&](uint32_t block) {
                    const uint32_t first = block * BlockTriangles;
                    const uint32_t last = std::min(first + BlockTriangles, primitive.TriangleCount);
                    body(block, first, last);
                });
            };
            // The earliest corner of each source vertex determines the serial
            // weld's render-vertex order. Store source handles directly in the
            // final corner array while finding these minima concurrently.
            for_each_triangle_block([&](uint32_t, uint32_t first, uint32_t last) {
                for (uint32_t triangle = first; triangle < last; ++triangle) {
                    const auto vertices = primitive_indices.TriangleAt(triangle);
                    for (uint32_t k = 0u; k < 3u; ++k) {
                        const uint32_t corner = triangle * 3u + k, source = vertices[k];
                        weld.CornerVertices[corner] = source;
                        std::atomic_ref entry{weld.SourceVertices[source - weld.VertexFirst]};
                        uint32_t prior = entry.load(std::memory_order_relaxed);
                        while (corner < prior && !entry.compare_exchange_weak(prior, corner, std::memory_order_relaxed)) {}
                    }
                }
            });
            std::vector<uint32_t> offsets(block_count + 1u);
            for_each_triangle_block([&](uint32_t block, uint32_t first, uint32_t last) {
                uint32_t count = 0u;
                for (uint32_t corner = first * 3u; corner < last * 3u; ++corner) {
                    const uint32_t source = weld.CornerVertices[corner];
                    count += uint32_t(weld.SourceVertices[source - weld.VertexFirst] == corner);
                }
                offsets[block] = count;
            });
            uint32_t total = 0u;
            for (uint32_t block = 0u; block < block_count; ++block) {
                const uint32_t count = offsets[block];
                offsets[block] = total;
                total += count;
            }
            offsets[block_count] = total;
            weld.Representative.resize(total);
            weld.Positions.resize(total);
            for_each_triangle_block([&](uint32_t block, uint32_t first, uint32_t last) {
                uint32_t render_vertex = offsets[block];
                for (uint32_t corner = first * 3u; corner < last * 3u; ++corner) {
                    const uint32_t source = weld.CornerVertices[corner];
                    if (weld.SourceVertices[source - weld.VertexFirst] != corner) continue;
                    weld.Representative[render_vertex] = corner;
                    const float *position = mesh.Positions + (mesh.PositionStride / sizeof(float)) * (source - mesh.VertexFirst);
                    weld.Positions[render_vertex] = {position[0], position[1], position[2]};
                    ++render_vertex;
                }
                assert(render_vertex == offsets[block + 1u]);
            });
            constexpr uint32_t VertexBlock{64u * 1024u};
            ParallelFor((total + VertexBlock - 1u) / VertexBlock, [&](uint32_t block) {
                const uint32_t last = std::min((block + 1u) * VertexBlock, total);
                for (uint32_t vertex = block * VertexBlock; vertex < last; ++vertex) {
                    const uint32_t corner = weld.Representative[vertex];
                    const uint32_t source = weld.CornerVertices[corner];
                    weld.SourceVertices[source - weld.VertexFirst] = vertex;
                }
            });
            for_each_triangle_block([&](uint32_t, uint32_t first, uint32_t last) {
                for (uint32_t corner = first * 3u; corner < last * 3u; ++corner)
                    weld.CornerVertices[corner] = weld.SourceVertices[weld.CornerVertices[corner] - weld.VertexFirst];
            });
        } else {
            weld.Representative.reserve(corner_count);
            weld.Positions.reserve(corner_count);
            for (uint32_t triangle = 0; triangle < primitive.TriangleCount; ++triangle) {
                const auto vertices = primitive_indices.TriangleAt(triangle);
                for (uint32_t c = 0; c < 3u; ++c) {
                    const uint32_t corner = triangle * 3u + c;
                    const uint32_t source_vertex = vertices[c];
                    auto &render_vertex = weld.SourceVertices[source_vertex - weld.VertexFirst];
                    if (render_vertex == ClusterLodInvalid) render_vertex = append_render_vertex(corner, source_vertex);
                    weld.CornerVertices[corner] = render_vertex;
                }
            }
        }
    } else {
        weld.Representative.reserve(corner_count);
        weld.Positions.reserve(corner_count);
        // A source-vertex key reduces to its first word, the vertex, which the key map then holds alone.
        std::vector<uint8_t> flat_face_triangles(source_vertex_only ? 0u : primitive.TriangleCount);
        for (uint32_t triangle = 0; triangle < flat_face_triangles.size(); ++triangle) {
            flat_face_triangles[triangle] = key.FlatFaceTriangle(triangle);
        }

        weld.Keys.Reset(source_vertex_only ? 1u : key.WordCount(), corner_count);
        std::array<uint32_t, MaxWeldKeyWords> words{};
        for (uint32_t triangle = 0; triangle < primitive.TriangleCount; ++triangle) {
            const auto vertices = primitive_indices.TriangleAt(triangle);
            for (uint32_t c = 0; c < 3u; ++c) {
                const uint32_t corner = triangle * 3u + c;
                const uint32_t source_vertex = vertices[c];
                if (source_vertex_only) words[0] = source_vertex;
                else key.Write(corner, source_vertex, flat_face_triangles[triangle], words);
                if (const auto *found = weld.Keys.Find(words.data())) {
                    weld.CornerVertices[corner] = *found;
                    continue;
                }
                weld.Keys.Insert(words.data(), append_render_vertex(corner, source_vertex));
            }
        }
    }

    const uint32_t weld_count = weld.VertexCount();
    std::array<bool, MaxWeldUvSets> active_uvs{};
    for (uint32_t uv = 0; uv < MaxWeldUvSets; ++uv)
        active_uvs[uv] = (primitive.Attributes & (MeshAttributeBit_TexCoord0 << uv)) && !mesh.Weld.CornerUvs[uv].empty();
    const bool active_tangents = (primitive.Attributes & MeshAttributeBit_Tangent) && !mesh.Weld.CornerTangents.empty();
    weld.AttributeCount = 3u + 2u * uint32_t(std::ranges::count(active_uvs, true)) +
        (active_tangents ? 4u : 0u) + (mesh.Weld.CornerColors.empty() ? 0u : 4u);
    weld.Attributes.resize(size_t(weld_count) * weld.AttributeCount);
    for (uint32_t v = 0; v < weld_count; ++v) {
        const uint32_t handle = key.Handle(weld.Representative[v]);
        float *attribute = &weld.Attributes[size_t(v) * weld.AttributeCount];
        // Flat-face curvature uses the shared vertex normal at geometric error scale.
        const auto normal = key.FlatFaceTriangle(weld.Representative[v] / 3u) ?
            mesh.Normals.VertexNormals[mesh.CornerVertices[first_index + weld.Representative[v]]] / ShadingAttributeScale :
            mesh.Normals[handle];
        *attribute++ = normal.x;
        *attribute++ = normal.y;
        *attribute++ = normal.z;
        for (uint32_t set = 0; set < MaxWeldUvSets; ++set) {
            if (!active_uvs[set]) continue;
            const auto uv = mesh.Weld.CornerUvs[set].Attribute[handle];
            *attribute++ = uv.x;
            *attribute++ = uv.y;
        }
        if (active_tangents) {
            const auto tangent = mesh.Weld.CornerTangents.Attribute[handle];
            const vec3 vector{tangent.x, tangent.y, tangent.z};
            const auto direction = Dot(vector, vector) > 1e-8f ? Normalize(vector) : vec3{};
            *attribute++ = direction.x;
            *attribute++ = direction.y;
            *attribute++ = direction.z;
            *attribute++ = tangent.w;
        }
        if (!mesh.Weld.CornerColors.empty()) {
            const auto color = mesh.Weld.CornerColors.Attribute[handle];
            *attribute++ = color.x;
            *attribute++ = color.y;
            *attribute++ = color.z;
            *attribute++ = color.w;
        }
        assert(attribute == &weld.Attributes[size_t(v + 1u) * weld.AttributeCount]);
    }

    // Cluster connectivity and consistent boundary locking both run over positions alone.
    weld.Remap.assign(weld_count, 0u);
    if (serial || weld_count < ParallelPositionRemapVertices) {
        meshopt_generatePositionRemap(weld.Remap.data(), weld.Positions.front().data(), weld_count, sizeof(weld.Positions.front()));
    } else {
        GeneratePositionRemap(weld.Remap, weld.Positions);
    }

    // The primitive extent scales every shading attribute consistently across group rebuilds.
    weld.Scale = meshopt_simplifyScale(weld.Positions.front().data(), weld_count, sizeof(weld.Positions.front()));

    // Distinct render keys at one position keep every side of the seam fixed at every level.
    weld.Locks.assign(weld_count, 0u);
    weld.SeamLocks.assign(weld_count, 0u);
    weld.Owners.resize(weld_count);
    for (uint32_t v = 0; v < weld_count; ++v) {
        const uint32_t canonical = weld.Remap[v];
        if (canonical != v) weld.SeamLocks[v] = weld.SeamLocks[canonical] = uint8_t(meshopt_SimplifyVertex_Lock);
    }
}

// One cluster the DAG is still working with, in weld-vertex indices.
struct WorkCluster {
    std::vector<uint32_t> Vertices;
    std::vector<uint32_t> Corners; // canonical corner handle per local vertex
    std::vector<uint8_t> LocalTriangles, PhysicalBoundaries;
    Bounds Sphere;
    uint32_t Refined{ClusterLodInvalid}; // the group this cluster was simplified from
    uint32_t Level0Id{ClusterLodInvalid};
    bool ConeSafe{};

    uint32_t TriangleCount() const { return uint32_t(LocalTriangles.size() / 3u); }
};

// Counts a group's edges without allocating per edge or sorting its corners.
struct GroupEdges {
    struct Entry {
        uint64_t Key{};
        uint32_t Count{}, OutputCount{}, Epoch{};
        bool Physical{}, Reversed{};
    };
    std::vector<Entry> Table;
    std::vector<uint32_t> Occupied;
    uint32_t Epoch{};

    void Reset(size_t edges) {
        const auto capacity = std::bit_ceil(std::max<size_t>(edges * 2u, 64u));
        if (Table.size() < capacity) {
            Table.assign(capacity, Entry{});
            Epoch = 0u;
        }
        if (++Epoch == 0u) {
            std::ranges::fill(Table, Entry{});
            Epoch = 1u;
        }
        Occupied.clear();
        Occupied.reserve(edges);
    }
    static uint64_t Hash(uint64_t key) {
        const uint64_t a = uint32_t(key), b = key >> 32u;
        const uint64_t mixed = (a * 0x9e3779b185ebca87ull) ^ (b * 0xc2b2ae3d27d4eb4full);
        return mixed ^ (mixed >> 32u);
    }
    Entry &Add(uint64_t key, bool physical, bool output = false, bool reversed = false) {
        const auto mask = Table.size() - 1u;
        for (auto i = Hash(key) & mask;; i = (i + 1u) & mask) {
            auto &entry = Table[i];
            if (entry.Epoch != Epoch) {
                entry = {.Key = key, .Count = uint32_t(!output), .OutputCount = uint32_t(output), .Epoch = Epoch, .Physical = physical, .Reversed = reversed};
                Occupied.push_back(uint32_t(i));
                return entry;
            }
            if (entry.Key == key) {
                if (output) ++entry.OutputCount;
                else ++entry.Count;
                return entry;
            }
        }
    }
    Entry &AddOutput(uint64_t key) { return Add(key, false, true); }
    const Entry *Find(uint64_t key) const {
        const auto mask = Table.size() - 1u;
        for (auto i = Hash(key) & mask;; i = (i + 1u) & mask) {
            const auto &entry = Table[i];
            if (entry.Epoch != Epoch) return nullptr;
            if (entry.Key == key) return &entry;
        }
    }
};

// Lock every weld vertex shared by two groups to preserve their boundary during simplification.
std::vector<std::vector<uint64_t>> LockBoundary(PrimitiveWeld &weld, const std::vector<WorkCluster> &clusters, const std::vector<std::vector<uint32_t>> &groups, bool serial) {
    constexpr uint8_t LockBit{uint8_t(meshopt_SimplifyVertex_Lock)};
    constexpr auto Relaxed{std::memory_order_relaxed};
    const bool parallel = !serial && groups.size() > 1u;
    // Parallel passes split the groups into at most 64 contiguous runs.
    const auto for_each_group = [&](auto &&function) {
        const auto chunks = parallel ? uint32_t(std::min<size_t>(groups.size(), 64u)) : 1u;
        const auto run = [&](uint32_t chunk) {
            const auto end = groups.size() * (chunk + 1u) / chunks;
            for (size_t group = groups.size() * chunk / chunks; group < end; ++group) function(group);
        };
        if (parallel) ParallelFor(chunks, run);
        else run(0u);
    };
    const auto for_each_vertex = [&](size_t group, auto &&function) {
        for (const auto member : groups[group])
            for (const auto vertex : clusters[member].Vertices) function(vertex);
    };
    // Only this level's vertices participate. Other levels may reference most
    // of the repair pool, so resetting that whole pool here repeats unrelated work.
    const auto lock_of = [&](uint32_t vertex) { return std::atomic_ref<uint8_t>{weld.Locks[vertex]}; };
    const auto owner_of = [&](uint32_t vertex) { return std::atomic_ref<uint32_t>{weld.Owners[vertex]}; };
    uint64_t level_vertices = 0u;
    for (const auto &group : groups)
        for (const auto member : group) level_vertices += clusters[member].Vertices.size();
    profile::RecordCounter("LodBoundaryVertices", level_vertices);
    profile::RecordCounter("LodBoundaryPoolVertices", weld.VertexCount());
    for (size_t group = 0u; group < groups.size(); ++group) {
        for_each_vertex(group, [&](uint32_t vertex) {
            const uint32_t canonical = weld.Remap[vertex];
            weld.Locks[canonical] = weld.SeamLocks[canonical];
            weld.Owners[canonical] = ClusterLodInvalid;
        });
    }

    // A material seam or a boundary against a terminal group is absent from this primitive's pending groups.
    // Its nonphysical open edges must still stay fixed.
    // The shared-vertex lock cannot discover that neighbor.
    // An edge is open exactly when its undirected key occurs once in the group.
    std::vector<std::vector<uint64_t>> artificial_edges(groups.size());
    const auto lock_group_edges = [&](GroupEdges &edges, size_t group_index) {
        const auto &group = groups[group_index];
        auto &artificial = artificial_edges[group_index];
        size_t count = 0u;
        for (const auto member : group) count += clusters[member].LocalTriangles.size();
        edges.Reset(count);
        for (const auto member : group) {
            const auto &cluster = clusters[member];
            for (uint32_t c = 0u; c < cluster.LocalTriangles.size(); ++c) {
                const uint32_t a = weld.Remap[cluster.Vertices[cluster.LocalTriangles[c]]];
                const uint32_t d = weld.Remap[cluster.Vertices[cluster.LocalTriangles[c / 3u * 3u + (c + 1u) % 3u]]];
                edges.Add(uint64_t(std::min(a, d)) << 32u | std::max(a, d), cluster.PhysicalBoundaries[c] != 0u, false, a > d);
            }
        }
        for (const auto slot : edges.Occupied) {
            const auto &entry = edges.Table[slot];
            if (entry.Count == 1u && !entry.Physical) {
                const uint32_t a = uint32_t(entry.Key >> 32u), d = uint32_t(entry.Key);
                // A unique undirected edge has one directed input occurrence.
                // Self-edges have a matching reverse and were never artificial.
                if (a != d) artificial.push_back(entry.Reversed ? (uint64_t(d) << 32u | a) : entry.Key);
                lock_of(a).fetch_or(LockBit, Relaxed);
                lock_of(d).fetch_or(LockBit, Relaxed);
            }
        }
        std::ranges::sort(artificial);
    };
    // Reuse the bounded edge table across chunks and levels on each worker.
    thread_local GroupEdges edges;
    for_each_group([&](size_t group) {
        for_each_vertex(group, [&](uint32_t vertex) {
            const uint32_t canonical = weld.Remap[vertex];
            const auto owner = owner_of(canonical);
            uint32_t expected = ClusterLodInvalid;
            if (!owner.compare_exchange_strong(expected, uint32_t(group), Relaxed) && expected != group) lock_of(canonical).fetch_or(LockBit, Relaxed);
        });
        lock_group_edges(edges, group);
    });

    for (size_t group = 0u; group < groups.size(); ++group) {
        for_each_vertex(group, [&](uint32_t vertex) {
            weld.Locks[vertex] = weld.Locks[weld.Remap[vertex]];
        });
    }
    return artificial_edges;
}

std::vector<std::vector<uint32_t>> PartitionClusters(const PrimitiveWeld &weld, const std::vector<WorkCluster> &clusters, const std::vector<uint32_t> &pending) {
    if (pending.size() <= PartitionSize) return {pending};

    std::vector<uint32_t> cluster_indices, cluster_counts(pending.size());
    size_t total_index_count = 0;
    for (const auto member : pending) total_index_count += clusters[member].Vertices.size();
    cluster_indices.reserve(total_index_count);
    for (size_t i = 0; i < pending.size(); ++i) {
        const auto &cluster = clusters[pending[i]];
        cluster_counts[i] = uint32_t(cluster.Vertices.size());
        for (const auto vertex : cluster.Vertices) cluster_indices.push_back(weld.Remap[vertex]);
    }

    std::vector<uint32_t> cluster_partition(pending.size());
    const auto partition_count = meshopt_partitionClusters(
        cluster_partition.data(), cluster_indices.data(), cluster_indices.size(), cluster_counts.data(), cluster_counts.size(),
        weld.Positions.front().data(), weld.VertexCount(), sizeof(weld.Positions.front()), PartitionSize
    );

    std::vector<std::vector<uint32_t>> groups(partition_count);
    for (auto &group : groups) group.reserve(PartitionSize + PartitionSize / 3);
    for (size_t i = 0; i < pending.size(); ++i) groups[cluster_partition[i]].push_back(pending[i]);
    return groups;
}

// Halves a group's triangle count while keeping its locked boundary and render seams.
// Returns error in mesh units without an edge-length limit.
std::vector<uint32_t> SimplifyGroup(const PrimitiveWeld &weld, const std::vector<uint32_t> &indices, size_t target_count, float *error) {
    if (target_count > indices.size()) return indices;

    std::array<float, 3u + 2u * MaxWeldUvSets + 4u + 4u> attribute_weights;
    attribute_weights.fill(weld.Scale * ShadingAttributeScale);
    constexpr uint32_t Options{meshopt_SimplifySparse | meshopt_SimplifyErrorAbsolute | meshopt_SimplifyPermissive};
    std::vector<uint32_t> lod(indices.size());
    lod.resize(meshopt_simplifyWithAttributes(
        lod.data(), indices.data(), indices.size(),
        weld.Positions.front().data(), weld.VertexCount(), sizeof(weld.Positions.front()),
        weld.Attributes.data(), sizeof(float) * weld.AttributeCount, attribute_weights.data(), weld.AttributeCount,
        weld.Locks.data(), target_count, FLT_MAX, Options, error
    ));
    return lod;
}

std::vector<WorkCluster> Clusterize(const PrimitiveWeld &weld, const std::vector<uint32_t> &indices) {
    const auto bound = meshopt_buildMeshletsBound(indices.size(), ClusterLodMaxVertices, ClusterLodMinTriangles);
    std::vector<meshopt_Meshlet> built(bound);
    std::vector<uint32_t> local_vertices(bound * ClusterLodMaxVertices);
    std::vector<uint8_t> local_triangles(bound * ClusterLodMaxTriangles * 3u);
    built.resize(meshopt_buildMeshletsFlex(
        built.data(), local_vertices.data(), local_triangles.data(), indices.data(), indices.size(),
        weld.Positions.front().data(), weld.VertexCount(), sizeof(weld.Positions.front()),
        ClusterLodMaxVertices, ClusterLodMinTriangles, ClusterLodMaxTriangles, ClusterConeWeight, ClusterSplitFactor
    ));

    std::vector<WorkCluster> clusters(built.size());
    for (size_t i = 0; i < built.size(); ++i) {
        const auto &meshlet = built[i];
        const auto vertices = std::span{local_vertices}.subspan(meshlet.vertex_offset, meshlet.vertex_count);
        const auto triangles = std::span{local_triangles}.subspan(meshlet.triangle_offset, size_t(meshlet.triangle_count) * 3u);
        clusters[i].Vertices.assign(vertices.begin(), vertices.end());
        clusters[i].LocalTriangles.assign(triangles.begin(), triangles.end());
    }
    return clusters;
}

// One group's own output, merged into the build in group order.
struct GroupScratch {
    std::vector<ClusterLodCluster> Clusters;
    std::vector<uint32_t> VertexCorners;
    std::vector<uint8_t> LocalTriangles;
    std::vector<uint32_t> MemberLevel0; // per member, the input cluster id or ClusterLodInvalid
    std::vector<WorkCluster> NewClusters;
    Bounds Sphere;
    bool Stuck{};
};

// Appends one cluster with indices local to `sink`.
void EmitCluster(auto &&sink, const PrimitiveWeld &weld, const WorkCluster &cluster, uint32_t primitive, uint32_t group) {
    const uint32_t vertex_count = uint32_t(cluster.Vertices.size());
    const uint32_t triangle_count = cluster.TriangleCount();
    assert(vertex_count <= ClusterLodMaxVertices && triangle_count <= ClusterLodMaxTriangles);

    const auto bounds = meshopt_computeMeshletBounds(
        cluster.Vertices.data(), cluster.LocalTriangles.data(), triangle_count,
        weld.Positions.front().data(), weld.VertexCount(), sizeof(weld.Positions.front())
    );
    sink.Clusters.push_back(ClusterLodCluster{
        .VertexOffset = uint32_t(sink.VertexCorners.size()),
        .VertexCount = uint32_t(vertex_count),
        .LocalTriangleOffset = uint32_t(sink.LocalTriangles.size()),
        .TriangleCount = triangle_count,
        .Primitive = primitive,
        .ConeAxisCutoff = PackCone(bounds, cluster.ConeSafe),
        .Center = std::bit_cast<vec3>(bounds.center),
        .Radius = bounds.radius,
        .GroupIndex = group,
        .RefinedGroup = cluster.Refined,
    });
    assert(cluster.Corners.size() == vertex_count);
    sink.VertexCorners.insert(sink.VertexCorners.end(), cluster.Corners.begin(), cluster.Corners.end());
    for (uint32_t c = 0u; c < cluster.LocalTriangles.size(); ++c)
        sink.LocalTriangles.push_back(cluster.LocalTriangles[c] | (cluster.PhysicalBoundaries[c] ? uint8_t(MeshletGeometryEncoding::PhysicalBoundaryBit) : 0u));
}

// Preserve physical-boundary provenance in the CPU reference builder too.
// Shared partition edges keep both endpoints locked, so every newly created
// open edge follows a physical boundary. Unchanged partition edges retain their
// identity in the geometric position domain, independently of shading wedges.
void ClassifyOutputBoundaries(const PrimitiveWeld &weld, const std::vector<uint64_t> &artificial, std::vector<WorkCluster> &output) {
    const auto edge = [&](const WorkCluster &cluster, uint32_t c) {
        return std::pair{weld.Remap[cluster.Vertices[cluster.LocalTriangles[c]]], weld.Remap[cluster.Vertices[cluster.LocalTriangles[c / 3u * 3u + (c + 1u) % 3u]]]};
    };
    const auto key = [](uint32_t a, uint32_t d) { return (uint64_t(a) << 32u) | d; };
    // LockBoundary already identified the input's open nonphysical edges.
    // Only output edges need a table to find newly exposed boundaries.
    thread_local GroupEdges edges;
    size_t output_count = 0u;
    for (const auto &cluster : output) output_count += cluster.LocalTriangles.size();
    edges.Reset(output_count);
    for (const auto &cluster : output)
        for (uint32_t c = 0u; c < cluster.LocalTriangles.size(); ++c) {
            const auto [a, d] = edge(cluster, c);
            edges.AddOutput(key(a, d));
        }
    for (auto &cluster : output) {
        cluster.PhysicalBoundaries.resize(cluster.LocalTriangles.size());
        for (uint32_t c = 0u; c < cluster.LocalTriangles.size(); ++c) {
            const auto [a, d] = edge(cluster, c);
            const auto *forward = edges.Find(key(a, d)), *reverse = edges.Find(key(d, a));
            cluster.PhysicalBoundaries[c] = forward->OutputCount == 1u && (!reverse || !reverse->OutputCount) &&
                !std::binary_search(artificial.begin(), artificial.end(), key(a, d));
        }
    }
}

// Each output vertex names the corner of its weld vertex's first member occurrence.
// This confines a coarse cluster to corners that its finest descendants name.
void AssignMemberCorners(const std::vector<WorkCluster> &clusters, const std::vector<uint32_t> &members, std::vector<WorkCluster> &output) {
    thread_local FlatKeyMap first_corners;
    size_t member_vertices = 0;
    for (const auto member : members) member_vertices += clusters[member].Vertices.size();
    first_corners.Reset(1u, uint32_t(member_vertices));
    for (const auto member : members) {
        const auto &cluster = clusters[member];
        for (uint32_t v = 0; v < cluster.Vertices.size(); ++v) {
            const uint32_t *vertex = &cluster.Vertices[v];
            if (!first_corners.Find(vertex)) first_corners.Insert(vertex, cluster.Corners[v]);
        }
    }
    for (auto &cluster : output) {
        cluster.Corners.resize(cluster.Vertices.size());
        for (size_t v = 0; v < cluster.Vertices.size(); ++v) cluster.Corners[v] = *first_corners.Find(&cluster.Vertices[v]);
    }
}

void RunGroup(GroupScratch &scratch, const PrimitiveWeld &weld, const std::vector<WorkCluster> &clusters, const std::vector<uint32_t> &members, const std::vector<uint64_t> &artificial, uint32_t primitive, uint32_t group) {
    std::vector<Bounds> member_bounds(members.size());
    std::vector<uint32_t> merged;
    size_t merged_size = 0;
    for (const auto member : members) merged_size += clusters[member].LocalTriangles.size();
    merged.reserve(merged_size);
    for (size_t i = 0; i < members.size(); ++i) {
        const auto &cluster = clusters[members[i]];
        member_bounds[i] = cluster.Sphere;
        for (const auto local : cluster.LocalTriangles) merged.push_back(cluster.Vertices[local]);
    }
    // Reuse merged member bounds to preserve monotonic containment across levels.
    scratch.Sphere = MergeBounds(member_bounds);

    const size_t target_size = size_t(float(merged.size() / 3u) * SimplifyRatio) * 3u;
    float error = 0.f;
    // Preserve single-triangle groups to keep every coarser level complete.
    const auto simplified = target_size == 0 ? merged : SimplifyGroup(weld, merged, target_size, &error);
    scratch.Stuck = float(simplified.size()) > float(merged.size()) * SimplifyThreshold;

    bool cone_safe = true;
    scratch.MemberLevel0.reserve(members.size());
    for (const auto member : members) {
        const auto &cluster = clusters[member];
        cone_safe &= cluster.ConeSafe;
        scratch.MemberLevel0.push_back(cluster.Level0Id);
        if (cluster.Level0Id == ClusterLodInvalid) EmitCluster(scratch, weld, cluster, primitive, group);
    }
    if (scratch.Stuck) {
        scratch.Sphere.Error = FLT_MAX; // A terminal group simplifies no further.
        return;
    }

    scratch.Sphere.Error = std::max(scratch.Sphere.Error, error);
    scratch.NewClusters = Clusterize(weld, simplified);
    ClassifyOutputBoundaries(weld, artificial, scratch.NewClusters);
    AssignMemberCorners(clusters, members, scratch.NewClusters);
    // Inherit group bounds and error to preserve conservative tests at the next level.
    for (auto &cluster : scratch.NewClusters) {
        cluster.Sphere = scratch.Sphere;
        cluster.Refined = group;
        cluster.ConeSafe = cone_safe;
    }
}

// Runs the DAG's level loop over one primitive's pending clusters and merges each level's groups in partition order.
// joining[l] lists the existing clusters that enter the pending pool at level l.
// A single remaining cluster forms a terminal group, and the loop ends once no later level has clusters to join.
// Returns the number of levels.
uint32_t BuildLevels(ClusterLodBuild &build, PrimitiveWeld &weld, std::vector<WorkCluster> &clusters, std::vector<uint32_t> &pending, std::span<const std::vector<uint32_t>> joining, uint32_t primitive, bool serial) {
    const auto terminal = [&] {
        const auto &cluster = clusters[pending.front()];
        const uint32_t group = uint32_t(build.Groups.size());
        build.Groups.push_back(ClusterLodGroup{
            .Center = cluster.Sphere.Center,
            .Radius = cluster.Sphere.Radius,
            .Error = FLT_MAX,
            .FirstCluster = uint32_t(build.GroupClusters.size()),
            .ClusterCount = 1u,
            .Primitive = primitive,
        });
        if (cluster.Level0Id != ClusterLodInvalid) {
            build.Level0Groups[cluster.Level0Id] = group;
            build.GroupClusters.push_back(cluster.Level0Id);
        } else {
            build.GroupClusters.push_back(build.Level0Count() + uint32_t(build.Clusters.size()));
            EmitCluster(build, weld, cluster, primitive, group);
        }
        pending.clear();
    };
    std::vector<GroupScratch> scratch;
    uint32_t depth = 0;
    for (;; ++depth) {
        if (depth < joining.size()) pending.insert(pending.end(), joining[depth].begin(), joining[depth].end());
        if (pending.size() <= 1) {
            if (depth + 1u >= joining.size()) break;
            if (pending.size() == 1u) terminal();
            continue;
        }
        const auto groups = [&] {
            const profile::CpuScope stage{"LodPartition"};
            return PartitionClusters(weld, clusters, pending);
        }();
        const auto artificial_edges = [&] {
            const profile::CpuScope stage{"LodBoundary"};
            return LockBoundary(weld, clusters, groups, serial);
        }();

        const uint32_t group_base = uint32_t(build.Groups.size());
        scratch.assign(groups.size(), GroupScratch{});
        const auto run = [&](uint32_t i) { RunGroup(scratch[i], weld, clusters, groups[i], artificial_edges[i], primitive, group_base + i); };
        {
            const profile::CpuScope stage{"LodSimplifyClusters"};
            if (serial) {
                for (uint32_t i = 0; i < groups.size(); ++i) run(i);
            } else {
                ParallelFor(uint32_t(groups.size()), run);
            }
        }

        // Groups merge in partition order, so the DAG never depends on which group finished first.
        const profile::CpuScope stage{"LodMerge"};
        pending.clear();
        for (uint32_t i = 0; i < groups.size(); ++i) {
            auto &group_scratch = scratch[i];
            const uint32_t cluster_base = build.Level0Count() + uint32_t(build.Clusters.size());
            const uint32_t vertex_base = uint32_t(build.VertexCorners.size());
            const uint32_t local_triangle_base = uint32_t(build.LocalTriangles.size());
            for (auto &cluster : group_scratch.Clusters) {
                cluster.VertexOffset += vertex_base;
                cluster.LocalTriangleOffset += local_triangle_base;
            }
            build.Groups.push_back(ClusterLodGroup{
                .Center = group_scratch.Sphere.Center,
                .Radius = group_scratch.Sphere.Radius,
                .Error = group_scratch.Sphere.Error,
                .FirstCluster = uint32_t(build.GroupClusters.size()),
                .ClusterCount = uint32_t(group_scratch.MemberLevel0.size()),
                .Primitive = primitive,
            });
            uint32_t emitted = 0;
            for (const auto level0 : group_scratch.MemberLevel0) {
                if (level0 != ClusterLodInvalid) {
                    build.Level0Groups[level0] = group_base + i;
                    build.GroupClusters.push_back(level0);
                } else {
                    build.GroupClusters.push_back(cluster_base + emitted++);
                }
            }
            build.Clusters.insert(build.Clusters.end(), group_scratch.Clusters.begin(), group_scratch.Clusters.end());
            build.VertexCorners.insert(build.VertexCorners.end(), group_scratch.VertexCorners.begin(), group_scratch.VertexCorners.end());
            build.LocalTriangles.insert(build.LocalTriangles.end(), group_scratch.LocalTriangles.begin(), group_scratch.LocalTriangles.end());
            for (auto &cluster : group_scratch.NewClusters) {
                pending.push_back(uint32_t(clusters.size()));
                clusters.push_back(std::move(cluster));
            }
        }
    }
    if (pending.size() == 1u) {
        terminal();
        ++depth;
    }
    // Workers release the clusters' storage, as they allocated it.
    constexpr size_t ReleaseBlock{1024u};
    ParallelFor(uint32_t((clusters.size() + ReleaseBlock - 1u) / ReleaseBlock), [&](uint32_t block) {
        for (auto i = block * ReleaseBlock; i < std::min(clusters.size(), (block + 1u) * ReleaseBlock); ++i) clusters[i] = {};
    });
    clusters.clear();
    return depth;
}

// Emits both record and group bounds so each span node remains conservative for frustum and error tests.
void CollectSpanRecords(
    const ClusterLodBuild &build, const ClusterLodMesh &mesh, uint32_t primitive, std::vector<Bounds> &records
) {
    records.clear();
    const auto &source = mesh.Primitives[primitive];
    const auto &range = build.PrimitiveRanges[primitive];
    const auto push = [&](const vec3 &center, float radius, const ClusterLodGroup &group) {
        records.push_back({.Center = center, .Radius = radius, .Error = group.Error});
        records.push_back({.Center = group.Center, .Radius = group.Radius, .Error = group.Error});
    };
    for (uint32_t k = 0; k < source.ClusterCount; ++k) {
        const auto &cluster = mesh.Clusters[source.FirstCluster + k];
        push(cluster.Center, cluster.Radius, build.Groups[build.Level0Groups[source.FirstCluster + k]]);
    }
    for (uint32_t c = 0; c < range.ClusterCount; ++c) {
        const auto &cluster = build.Clusters[range.FirstCluster + c];
        push(cluster.Center, cluster.Radius, build.Groups[cluster.GroupIndex]);
    }
}

// Builds a span tree whose surviving record runs retain ascending order after bounds and error pruning.
void BuildSpanTree(
    ClusterLodBuild &build, const ClusterLodMesh &mesh, uint32_t primitive, uint32_t first_record,
    std::vector<Bounds> &records, std::vector<uint32_t> &row, std::vector<uint32_t> &next
) {
    auto &range = build.PrimitiveRanges[primitive];
    const uint32_t level0_count = mesh.Primitives[primitive].ClusterCount;
    const uint32_t count = level0_count + range.ClusterCount;
    if (count == 0) {
        range.RootNode = ClusterLodInvalid;
        range.FinestNode = ClusterLodInvalid;
        return;
    }
    // A pinned instance draws the original-geometry prefix whole, which one never-pruned leaf covers.
    range.FinestNode = uint32_t(build.Nodes.size());
    build.Nodes.push_back(LodNode{
        .Error = std::numeric_limits<float>::infinity(),
        .FirstMeshlet = first_record,
        .MeshletCount = level0_count,
    });

    CollectSpanRecords(build, mesh, primitive, records);
    row.clear();
    for (uint32_t i = 0; i < count; i += ClusterLodSpanLeafRecords) {
        const uint32_t span = std::min(ClusterLodSpanLeafRecords, count - i);
        const auto bounds = MergeSpanBounds(std::span{records}.subspan(size_t{i} * 2u, size_t{span} * 2u));
        row.push_back(uint32_t(build.Nodes.size()));
        build.Nodes.push_back(LodNode{
            .Center = bounds.Center,
            .Radius = bounds.Radius,
            .Error = bounds.Error,
            .FirstMeshlet = first_record + i,
            .MeshletCount = span,
        });
    }

    uint32_t depth = 0;
    std::vector<Bounds> children;
    while (row.size() > 1) {
        next.clear();
        for (size_t i = 0; i < row.size(); i += ClusterLodSpanNodeWidth) {
            const uint32_t span = uint32_t(std::min(size_t{ClusterLodSpanNodeWidth}, row.size() - i));
            children.clear();
            for (uint32_t c = 0; c < span; ++c) {
                const auto &child = build.Nodes[row[i + c]];
                children.push_back({.Center = child.Center, .Radius = child.Radius, .Error = child.Error});
            }
            const auto bounds = MergeSpanBounds(children);
            const auto &first = build.Nodes[row[i]];
            const auto &last = build.Nodes[row[i + span - 1]];
            next.push_back(uint32_t(build.Nodes.size()));
            build.Nodes.push_back(LodNode{
                .Center = bounds.Center,
                .Radius = bounds.Radius,
                .Error = bounds.Error,
                .FirstMeshlet = first.FirstMeshlet,
                .MeshletCount = last.FirstMeshlet + last.MeshletCount - first.FirstMeshlet,
                .ChildOffset = row[i],
                .ChildCount = span,
            });
        }
        row.swap(next);
        ++depth;
    }
    range.RootNode = row.front();
    build.NodeDepth = std::max(build.NodeDepth, depth);
}

// Builds each primitive's span tree over its own run of records, which follow primitive order.
void BuildSpanTrees(ClusterLodBuild &build, const ClusterLodMesh &mesh) {
    std::vector<Bounds> records;
    std::vector<uint32_t> row, next;
    uint32_t first_record = 0;
    for (uint32_t p = 0; p < mesh.Primitives.size(); ++p) {
        BuildSpanTree(build, mesh, p, first_record, records, row, next);
        first_record += mesh.Primitives[p].ClusterCount + build.PrimitiveRanges[p].ClusterCount;
    }
}
} // namespace

// A mixed-normal cluster stores the never-culls cutoff, so the cone test needs no separate flag.
uint32_t PackCone(const meshopt_Bounds &bounds, bool cone_cull_safe) {
    return uint32_t(uint8_t(bounds.cone_axis_s8[0])) |
        uint32_t(uint8_t(bounds.cone_axis_s8[1])) << 8u |
        uint32_t(uint8_t(bounds.cone_axis_s8[2])) << 16u |
        uint32_t(uint8_t(cone_cull_safe ? bounds.cone_cutoff_s8 : int8_t{127})) << 24u;
}

ClusterLodBuild BuildClusterLod(const ClusterLodMesh &mesh, bool serial) {
    ClusterLodBuild build{.Level0Groups = std::vector<uint32_t>(mesh.Clusters.size(), ClusterLodInvalid)};
    build.PrimitiveRanges.reserve(mesh.Primitives.size());

    PrimitiveWeld weld;
    std::vector<WorkCluster> clusters;
    std::vector<uint32_t> pending;
    for (uint32_t primitive = 0; primitive < mesh.Primitives.size(); ++primitive) {
        const auto &range = mesh.Primitives[primitive];
        ClusterLodPrimitiveRange primitive_range{
            .FirstCluster = uint32_t(build.Clusters.size()),
            .FirstGroup = uint32_t(build.Groups.size()),
        };
        if (range.ClusterCount == 0u) {
            build.PrimitiveRanges.push_back(primitive_range);
            continue;
        }

        BuildWeld(mesh, range, weld, serial);
        primitive_range.SimplifyScale = weld.Scale;
        clusters.assign(range.ClusterCount, WorkCluster{});
        pending.resize(range.ClusterCount);
        const CornerWeldKey key{mesh.Weld, range.FirstTriangle * 3u};
        const auto initialize_cluster = [&](uint32_t i) {
            pending[i] = i;
            const auto &source = mesh.Clusters[range.FirstCluster + i];
            auto &cluster = clusters[i];
            cluster.Vertices.resize(source.VertexCount);
            const auto corners = mesh.SourceVertexCorners.subspan(source.FirstVertex, source.VertexCount);
            cluster.Corners.assign(corners.begin(), corners.end());
            std::vector<uint8_t> flat_vertices(source.VertexCount);
            cluster.LocalTriangles.resize(size_t(source.TriangleCount) * 3u);
            cluster.PhysicalBoundaries.resize(cluster.LocalTriangles.size());
            for (uint32_t c = 0; c < source.TriangleCount * 3u; ++c) {
                const uint8_t local = mesh.SourceLocalTriangles[source.FirstLocalTriangle + c] & uint8_t(MeshletGeometryEncoding::LocalIndexMask);
                if (local >= source.VertexCount) throw std::runtime_error(std::format("LOD input has invalid local vertex: primitive {}, cluster {}, corner {}, byte {}, local {}, vertex count {}, first vertex {}, first triangle byte {}, triangle count {}.", primitive, i, c, uint32_t(mesh.SourceLocalTriangles[source.FirstLocalTriangle + c]), uint32_t(local), source.VertexCount, source.FirstVertex, source.FirstLocalTriangle, source.TriangleCount));
                cluster.LocalTriangles[c] = local;
                cluster.PhysicalBoundaries[c] = (mesh.SourceLocalTriangles[source.FirstLocalTriangle + c] & uint8_t(MeshletGeometryEncoding::PhysicalBoundaryBit)) != 0u;
                flat_vertices[local] = (mesh.SourceLocalTriangles[source.FirstLocalTriangle + c / 3u * 3u] & uint8_t(MeshletGeometryEncoding::FlatTriangleBit)) != 0u;
            }
            for (uint32_t v = 0; v < source.VertexCount; ++v) {
                const uint32_t corner = cluster.Corners[v];
                const uint32_t vertex = mesh.CornerVertices.Vertices[corner];
                if (!weld.SourceVertices.empty()) {
                    cluster.Vertices[v] = weld.SourceVertices[vertex - weld.VertexFirst];
                } else {
                    std::array<uint32_t, MaxWeldKeyWords> words;
                    key.WriteHandle(corner, vertex, flat_vertices[v], words);
                    const auto *found = weld.Keys.Find(words.data());
                    if (!found) throw std::runtime_error(std::format("LOD cluster key is absent from source: primitive {}, cluster {}, local vertex {}, corner {}, vertex {}, flat {}, class {}, identity {}, mode {}, words {}, input corners {}.", primitive, i, v, corner, vertex, uint32_t(flat_vertices[v]), words[1], words[2], mesh.Weld.CornerClassMode, key.WordCount(), range.TriangleCount * 3u));
                    cluster.Vertices[v] = *found;
                }
            }
            cluster.Sphere = Bounds{.Center = source.Center, .Radius = source.Radius};
            cluster.Level0Id = range.FirstCluster + i;
            cluster.ConeSafe = source.ConeCullSafe;
        };
        if (serial || range.ClusterCount < 1024u) {
            for (uint32_t i = 0u; i < range.ClusterCount; ++i) initialize_cluster(i);
        } else {
            constexpr uint32_t ClustersPerBlock{2048u};
            std::mutex failure_mutex;
            std::exception_ptr failure;
            ParallelFor((range.ClusterCount + ClustersPerBlock - 1u) / ClustersPerBlock, [&](uint32_t block) {
                try {
                    const uint32_t last = std::min((block + 1u) * ClustersPerBlock, range.ClusterCount);
                    for (uint32_t i = block * ClustersPerBlock; i < last; ++i) initialize_cluster(i);
                } catch (...) {
                    std::lock_guard lock{failure_mutex};
                    if (!failure) failure = std::current_exception();
                }
            });
            if (failure) std::rethrow_exception(failure);
        }
        const uint32_t depth = BuildLevels(build, weld, clusters, pending, {}, primitive, serial);

        primitive_range.ClusterCount = uint32_t(build.Clusters.size()) - primitive_range.FirstCluster;
        primitive_range.GroupCount = uint32_t(build.Groups.size()) - primitive_range.FirstGroup;
        build.PrimitiveRanges.push_back(primitive_range);
        build.LevelCount = std::max(build.LevelCount, depth);
    }

    BuildSpanTrees(build, mesh);
    return build;
}

ClusterLodBuild RebuildClusterLod(const ClusterLodMesh &mesh, std::span<const uint32_t> levels, float scale) {
    assert(mesh.Primitives.size() == 1u && !mesh.Clusters.empty() && levels.size() == mesh.Clusters.size());
    ClusterLodBuild build{.Level0Groups = std::vector<uint32_t>(mesh.Clusters.size(), ClusterLodInvalid)};
    PrimitiveWeld weld;
    {
        const profile::CpuScope stage{"LodRepairWeld"};
        BuildWeld(mesh, mesh.Primitives.front(), weld, false);
    }
    weld.Scale = scale;
    // Each cluster's local vertices are the distinct weld vertices of its corners, in first-use order.
    // Each local vertex names the corner of its first use, and an open-addressed table finds it by weld vertex.
    std::vector<WorkCluster> clusters(mesh.Clusters.size());
    std::vector<uint32_t> first_corners(clusters.size() + 1u);
    for (uint32_t i = 0; i < clusters.size(); ++i) first_corners[i + 1u] = first_corners[i] + mesh.Clusters[i].TriangleCount * 3u;
    assert(first_corners.back() == mesh.Primitives.front().TriangleCount * 3u);
    constexpr uint32_t ClustersPerBlock{16u};
    ParallelFor((uint32_t(clusters.size()) + ClustersPerBlock - 1u) / ClustersPerBlock, [&](uint32_t block) {
        const auto last = std::min(uint32_t(clusters.size()), (block + 1u) * ClustersPerBlock);
        std::array<uint8_t, 256> slots;
        for (uint32_t i = block * ClustersPerBlock; i < last; ++i) {
            const auto &source = mesh.Clusters[i];
            auto &cluster = clusters[i];
            cluster.LocalTriangles.resize(size_t(source.TriangleCount) * 3u);
            cluster.PhysicalBoundaries.resize(cluster.LocalTriangles.size());
            slots.fill(0xffu);
            for (uint32_t c = 0; c < cluster.LocalTriangles.size(); ++c) {
                const uint32_t corner = first_corners[i] + c;
                const uint32_t vertex = weld.CornerVertices[corner];
                auto slot = (vertex * 0x9e3779b1u) >> 24u;
                while (slots[slot] != 0xffu && cluster.Vertices[slots[slot]] != vertex) slot = (slot + 1u) & 255u;
                if (slots[slot] == 0xffu) {
                    slots[slot] = uint8_t(cluster.Vertices.size());
                    cluster.Vertices.push_back(vertex);
                    cluster.Corners.push_back(weld.CanonicalCorner(corner));
                }
                cluster.LocalTriangles[c] = slots[slot];
                cluster.PhysicalBoundaries[c] = (mesh.SourceLocalTriangles[source.FirstLocalTriangle + c] & uint8_t(MeshletGeometryEncoding::PhysicalBoundaryBit)) != 0u;
            }
            cluster.Sphere = {.Center = source.Center, .Radius = source.Radius, .Error = source.Error};
            // Inputs are existing records, so the loop emits none of them.
            cluster.Level0Id = i;
            cluster.ConeSafe = source.ConeCullSafe;
        }
    });
    std::vector<std::vector<uint32_t>> joining(*std::ranges::max_element(levels) + 1u);
    for (uint32_t i = 0; i < clusters.size(); ++i) joining[levels[i]].push_back(i);
    std::vector<uint32_t> pending;
    {
        const profile::CpuScope stage{"LodRepairLevels"};
        build.LevelCount = BuildLevels(build, weld, clusters, pending, joining, 0u, false);
    }
    return build;
}
