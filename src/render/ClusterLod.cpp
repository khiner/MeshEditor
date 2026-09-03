#include "render/ClusterLod.h"

#include "FlatKeyMap.h"
#include "Parallel.h"
#include "gpu/MeshletGeometryEncoding.h"

#include "meshoptimizer.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

namespace {
constexpr size_t PartitionSize{ClusterLodPartitionSize};
constexpr float SimplifyRatio{0.5f};
constexpr float SimplifyThreshold{0.85f};
constexpr float SloppyErrorFactor{2.f};
constexpr float NormalWeight{0.25f};
constexpr float ClusterConeWeight{0.5f};
constexpr float ClusterSplitFactor{2.f};
// A wider tree reduces depth while increasing the granularity of pruned record runs.
constexpr uint32_t SpanNodeWidth{64};
constexpr float SpanNodeSlack{1.f + 1e-5f};
// Use nested parallelism only when one weld can occupy the machine independently.
constexpr uint32_t ParallelPositionRemapVertices{256u * 1024u};

using Clock = std::chrono::steady_clock;

double MillisecondsSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

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

// A mixed-normal cluster stores the never-culls cutoff, so the cone test needs no separate flag.
uint32_t PackCone(const meshopt_Bounds &bounds, bool cone_cull_safe) {
    return uint32_t(uint8_t(bounds.cone_axis_s8[0])) |
        uint32_t(uint8_t(bounds.cone_axis_s8[1])) << 8u |
        uint32_t(uint8_t(bounds.cone_axis_s8[2])) << 16u |
        uint32_t(uint8_t(cone_cull_safe ? bounds.cone_cutoff_s8 : int8_t{127})) << 24u;
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
    Bounds result{
        .Center = std::bit_cast<vec3>(merged.center),
        .Radius = merged.radius,
        .Error = 0.f,
    };
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
    std::vector<float> Normals; // three per weld vertex
    std::vector<uint32_t> Remap; // canonical weld vertex sharing a position
    std::vector<uint8_t> Locks;
    float Scale{}; // Extent factor from meshopt_simplifyScale, converting normalized weights to mesh units.

    uint32_t VertexCount() const { return uint32_t(Representative.size()); }
};

// Uses the shared render-equivalence key for both source and coarse cluster vertices.
void BuildWeld(const ClusterLodMesh &mesh, const ClusterLodPrimitive &primitive, PrimitiveWeld &weld, bool serial) {
    const uint32_t first_index = primitive.FirstTriangle * 3u;
    const uint32_t corner_count = primitive.TriangleCount * 3u;
    const auto primitive_indices = mesh.CornerVertices.subspan(first_index, corner_count);
    const CornerWeldKey key{mesh.Weld, first_index};

    weld.CornerVertices.assign(corner_count, 0u);
    weld.Representative.clear();
    weld.Positions.clear();
    weld.Representative.reserve(corner_count);
    weld.Positions.reserve(corner_count);
    const auto append_render_vertex = [&](uint32_t corner) {
        const uint32_t render_vertex = uint32_t(weld.Representative.size());
        weld.CornerVertices[corner] = render_vertex;
        weld.Representative.push_back(corner);
        const float *position = mesh.Positions + (mesh.PositionStride / sizeof(float)) * primitive_indices[corner];
        weld.Positions.push_back({position[0], position[1], position[2]});
        return render_vertex;
    };
    const uint32_t max_source_vertex = std::ranges::max(primitive_indices);
    const bool dense_source_vertices = size_t(max_source_vertex) + 1u <= size_t(corner_count) * 2u;
    const bool uniform_face = mesh.Weld.CornerClassOffset == uint32_t(CornerClassEncoding::UniformFaceOffset);
    const bool source_vertex_only = key.WordCount() == 2u && mesh.Weld.CornerClasses.empty() &&
        mesh.Weld.CustomCornerMasks.empty() && (!uniform_face || !mesh.Weld.MorphShadingAuthored);
    if (source_vertex_only && dense_source_vertices) {
        std::vector<uint32_t> source_vertices(size_t(max_source_vertex) + 1u, ClusterLodInvalid);
        for (uint32_t corner = 0; corner < corner_count; ++corner) {
            auto &render_vertex = source_vertices[primitive_indices[corner]];
            if (render_vertex == ClusterLodInvalid) render_vertex = append_render_vertex(corner);
            weld.CornerVertices[corner] = render_vertex;
        }
    } else {
        std::vector<uint8_t> flat_face_triangles(primitive.TriangleCount);
        for (uint32_t triangle = 0; triangle < primitive.TriangleCount; ++triangle) {
            flat_face_triangles[triangle] = key.FlatFaceTriangle(triangle);
        }

        FlatKeyMap welded;
        welded.Reset(key.WordCount(), corner_count);
        std::array<uint32_t, MaxWeldKeyWords> words{};
        for (uint32_t corner = 0; corner < corner_count; ++corner) {
            if (key.WeldsAlone(corner)) {
                append_render_vertex(corner);
                continue;
            }
            key.Write(corner, primitive_indices[corner], flat_face_triangles[corner / 3u], words);
            if (const auto *found = welded.Find(words.data())) {
                weld.CornerVertices[corner] = *found;
                continue;
            }
            welded.Insert(words.data(), append_render_vertex(corner));
        }
    }

    const uint32_t weld_count = weld.VertexCount();
    weld.Normals.assign(size_t(weld_count) * 3u, 0.f);
    if (!mesh.CornerNormals.empty()) {
        for (uint32_t v = 0; v < weld_count; ++v) {
            const vec3 normal = mesh.CornerNormals[first_index + weld.Representative[v]];
            weld.Normals[size_t(v) * 3u] = normal.x;
            weld.Normals[size_t(v) * 3u + 1u] = normal.y;
            weld.Normals[size_t(v) * 3u + 2u] = normal.z;
        }
    } else {
        // Area weighting falls out of accumulating the unnormalized triangle normal at every corner.
        for (uint32_t triangle = 0; triangle < primitive.TriangleCount; ++triangle) {
            const auto &a = weld.Positions[weld.CornerVertices[triangle * 3u]];
            const auto &b = weld.Positions[weld.CornerVertices[triangle * 3u + 1u]];
            const auto &c = weld.Positions[weld.CornerVertices[triangle * 3u + 2u]];
            const float ab[3]{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
            const float ac[3]{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
            const float normal[3]{
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            };
            for (uint32_t c2 = 0; c2 < 3u; ++c2) {
                const size_t vertex = weld.CornerVertices[triangle * 3u + c2];
                for (uint32_t k = 0; k < 3u; ++k) weld.Normals[vertex * 3u + k] += normal[k];
            }
        }
        for (uint32_t v = 0; v < weld_count; ++v) {
            float *normal = &weld.Normals[size_t(v) * 3u];
            const float length = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
            if (length > 0.f) {
                for (uint32_t k = 0; k < 3u; ++k) normal[k] /= length;
            }
        }
    }

    // Cluster connectivity and consistent boundary locking both run over positions alone.
    weld.Remap.assign(weld_count, 0u);
    if (serial || weld_count < ParallelPositionRemapVertices) {
        meshopt_generatePositionRemap(weld.Remap.data(), weld.Positions.front().data(), weld_count, sizeof(weld.Positions.front()));
    } else {
        GeneratePositionRemap(weld.Remap, weld.Positions);
    }

    // Scale normalized normal weights by primitive extent to compare them with mesh-unit position deltas.
    weld.Scale = meshopt_simplifyScale(weld.Positions.front().data(), weld_count, sizeof(weld.Positions.front()));

    // Protect first-UV discontinuities from permissive simplification.
    weld.Locks.assign(weld_count, 0u);
    if (const auto uvs = mesh.Weld.CornerUvs[0]; !uvs.empty()) {
        for (uint32_t v = 0; v < weld_count; ++v) {
            const uint32_t canonical = weld.Remap[v];
            if (canonical == v) continue;
            if (uvs[first_index + weld.Representative[v]] != uvs[first_index + weld.Representative[canonical]]) {
                weld.Locks[v] |= uint8_t(meshopt_SimplifyVertex_Protect);
            }
        }
    }
}

// One cluster the DAG is still working with, in weld-vertex indices.
struct WorkCluster {
    std::vector<uint32_t> Vertices;
    std::vector<uint8_t> LocalTriangles;
    Bounds Sphere;
    uint32_t Refined{ClusterLodInvalid}; // the group this cluster was simplified from
    uint32_t Level0Id{ClusterLodInvalid};
    bool ConeSafe{};

    uint32_t TriangleCount() const { return uint32_t(LocalTriangles.size() / 3u); }
};

// Lock every weld vertex shared by two groups to preserve their boundary during simplification.
void LockBoundary(PrimitiveWeld &weld, const std::vector<WorkCluster> &clusters, const std::vector<std::vector<uint32_t>> &groups) {
    constexpr uint8_t SeenBit{1u << 7u};
    constexpr uint8_t PersistentBits{uint8_t(meshopt_SimplifyVertex_Protect)};
    const auto for_each_vertex = [&](auto &&function) {
        for (const auto &group : groups)
            for (const auto member : group)
                for (const auto vertex : clusters[member].Vertices) function(vertex);
    };
    for_each_vertex([&](uint32_t vertex) {
        weld.Locks[vertex] &= PersistentBits;
        weld.Locks[weld.Remap[vertex]] &= PersistentBits;
    });

    for (const auto &group : groups) {
        // A vertex already used by a prior group belongs to a boundary.
        for (const auto member : group) {
            for (const auto vertex : clusters[member].Vertices) {
                const uint32_t canonical = weld.Remap[vertex];
                weld.Locks[canonical] |= uint8_t(weld.Locks[canonical] >> 7u);
            }
        }
        for (const auto member : group) {
            for (const auto vertex : clusters[member].Vertices) weld.Locks[weld.Remap[vertex]] |= SeenBit;
        }
    }

    for_each_vertex([&](uint32_t vertex) {
        const uint32_t canonical = weld.Remap[vertex];
        weld.Locks[vertex] = uint8_t((weld.Locks[canonical] & uint8_t(meshopt_SimplifyVertex_Lock)) | (weld.Locks[vertex] & PersistentBits));
    });
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

// The sloppy simplifier reaches targets regular simplification cannot, at a cost in appearance.
// It reads neither sparsity nor absolute error, so the group's vertices deindex into a subset first.
struct SloppyVertex {
    std::array<float, 3> Position;
    uint32_t Id;
};

void SimplifySloppy(std::vector<uint32_t> &lod, const PrimitiveWeld &weld, const std::vector<uint32_t> &indices, size_t target_count, float *error) {
    std::vector<SloppyVertex> subset(indices.size());
    std::vector<uint8_t> subset_locks(indices.size());
    lod.resize(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        const auto &position = weld.Positions[indices[i]];
        subset[i] = SloppyVertex{.Position = position, .Id = indices[i]};
        subset_locks[i] = weld.Locks[indices[i]];
        lod[i] = uint32_t(i);
    }
    lod.resize(meshopt_simplifySloppy(
        lod.data(), lod.data(), lod.size(), subset.front().Position.data(), subset.size(), sizeof(SloppyVertex),
        subset_locks.data(), target_count, FLT_MAX, error
    ));
    *error *= meshopt_simplifyScale(subset.front().Position.data(), subset.size(), sizeof(SloppyVertex));
    for (auto &index : lod) index = subset[index].Id;
}

// Halves a group's triangle count while protecting its boundary and UV discontinuities.
// Returns error in mesh units without an edge-length limit.
std::vector<uint32_t> SimplifyGroup(const PrimitiveWeld &weld, const std::vector<uint32_t> &indices, size_t target_count, float *error) {
    if (target_count > indices.size()) return indices;

    const float normal_weight = NormalWeight * weld.Scale;
    const float attribute_weights[3]{normal_weight, normal_weight, normal_weight};
    constexpr uint32_t Options{meshopt_SimplifySparse | meshopt_SimplifyErrorAbsolute | meshopt_SimplifyPermissive};
    std::vector<uint32_t> lod(indices.size());
    lod.resize(meshopt_simplifyWithAttributes(
        lod.data(), indices.data(), indices.size(),
        weld.Positions.front().data(), weld.VertexCount(), sizeof(weld.Positions.front()),
        weld.Normals.data(), sizeof(float) * 3u, attribute_weights, 3u,
        weld.Locks.data(), target_count, FLT_MAX, Options, error
    ));
    if (lod.size() > target_count) {
        SimplifySloppy(lod, weld, indices, target_count, error);
        *error *= SloppyErrorFactor;
    }
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
    uint32_t Triangles{};
    float RadiusSum{};
    bool Stuck{};
    double SimplifyMs{}, ClusterizeMs{}, EmitMs{};
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
    for (const auto vertex : cluster.Vertices) sink.VertexCorners.push_back(weld.Representative[vertex]);
    sink.LocalTriangles.insert(sink.LocalTriangles.end(), cluster.LocalTriangles.begin(), cluster.LocalTriangles.end());
}

void RunGroup(GroupScratch &scratch, const PrimitiveWeld &weld, const std::vector<WorkCluster> &clusters, const std::vector<uint32_t> &members, uint32_t primitive, uint32_t group) {
    std::vector<Bounds> member_bounds(members.size());
    std::vector<uint32_t> merged;
    size_t merged_size = 0;
    for (const auto member : members) merged_size += clusters[member].LocalTriangles.size();
    merged.reserve(merged_size);
    for (size_t i = 0; i < members.size(); ++i) {
        const auto &cluster = clusters[members[i]];
        member_bounds[i] = cluster.Sphere;
        for (const auto local : cluster.LocalTriangles) merged.push_back(cluster.Vertices[local]);
        scratch.Triangles += cluster.TriangleCount();
        scratch.RadiusSum += cluster.Sphere.Radius;
    }
    // Reuse merged member bounds to preserve monotonic containment across levels.
    scratch.Sphere = MergeBounds(member_bounds);

    const size_t target_size = size_t(float(merged.size() / 3u) * SimplifyRatio) * 3u;
    const auto simplify_start = Clock::now();
    float error = 0.f;
    // Preserve single-triangle groups to keep every coarser level complete.
    const auto simplified = target_size == 0 ? merged : SimplifyGroup(weld, merged, target_size, &error);
    scratch.SimplifyMs = MillisecondsSince(simplify_start);
    scratch.Stuck = float(simplified.size()) > float(merged.size()) * SimplifyThreshold;

    bool cone_safe = true;
    scratch.MemberLevel0.reserve(members.size());
    const auto emit_start = Clock::now();
    for (const auto member : members) {
        const auto &cluster = clusters[member];
        cone_safe &= cluster.ConeSafe;
        scratch.MemberLevel0.push_back(cluster.Level0Id);
        if (cluster.Level0Id == ClusterLodInvalid) EmitCluster(scratch, weld, cluster, primitive, group);
    }
    scratch.EmitMs = MillisecondsSince(emit_start);
    if (scratch.Stuck) {
        scratch.Sphere.Error = FLT_MAX; // A terminal group simplifies no further.
        return;
    }

    scratch.Sphere.Error = std::max(scratch.Sphere.Error, error);
    const auto clusterize_start = Clock::now();
    scratch.NewClusters = Clusterize(weld, simplified);
    scratch.ClusterizeMs = MillisecondsSince(clusterize_start);
    // Inherit group bounds and error to preserve conservative tests at the next level.
    for (auto &cluster : scratch.NewClusters) {
        cluster.Sphere = scratch.Sphere;
        cluster.Refined = group;
        cluster.ConeSafe = cone_safe;
    }
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
        for (size_t i = 0; i < row.size(); i += SpanNodeWidth) {
            const uint32_t span = uint32_t(std::min(size_t{SpanNodeWidth}, row.size() - i));
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

ClusterLodBuild BuildClusterLod(const ClusterLodMesh &mesh, bool serial) {
    const auto total_start = Clock::now();

    ClusterLodBuild build;
    build.Level0Groups.assign(mesh.Clusters.size(), ClusterLodInvalid);
    build.PrimitiveRanges.reserve(mesh.Primitives.size());

    PrimitiveWeld weld;
    std::vector<WorkCluster> clusters;
    std::vector<uint32_t> pending;
    std::vector<GroupScratch> scratch;
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

        const auto weld_start = Clock::now();
        BuildWeld(mesh, range, weld, serial);
        build.Stats.WeldMs += MillisecondsSince(weld_start);

        const auto level0_start = Clock::now();
        clusters.assign(range.ClusterCount, WorkCluster{});
        pending.resize(range.ClusterCount);
        std::iota(pending.begin(), pending.end(), 0u);
        for (uint32_t i = 0; i < range.ClusterCount; ++i) {
            const auto &source = mesh.Clusters[range.FirstCluster + i];
            auto &cluster = clusters[i];
            cluster.Vertices.resize(source.VertexCount);
            for (uint32_t v = 0; v < source.VertexCount; ++v) {
                const uint32_t corner = mesh.SourceVertexCorners[source.FirstVertex + v] & uint32_t(MeshletGeometryEncoding::CornerMask);
                assert(corner < range.TriangleCount * 3u);
                cluster.Vertices[v] = weld.CornerVertices[corner];
            }
            cluster.LocalTriangles.resize(size_t(source.TriangleCount) * 3u);
            for (uint32_t c = 0; c < source.TriangleCount * 3u; ++c) {
                const uint8_t local = mesh.SourceLocalTriangles[source.FirstLocalTriangle + c] & uint8_t(MeshletGeometryEncoding::LocalIndexMask);
                assert(local < source.VertexCount);
                cluster.LocalTriangles[c] = local;
            }
            cluster.Sphere = Bounds{
                .Center = source.Center,
                .Radius = source.Radius,
                .Error = 0.f,
            };
            cluster.Level0Id = range.FirstCluster + i;
            cluster.ConeSafe = source.ConeCullSafe;
        }
        build.Stats.Level0Ms += MillisecondsSince(level0_start);

        uint32_t depth = 0;
        while (pending.size() > 1) {
            const auto level_start = Clock::now();
            if (build.Stats.Levels.size() <= depth) build.Stats.Levels.emplace_back();
            auto &level_stats = build.Stats.Levels[depth];

            const auto partition_start = Clock::now();
            const auto groups = PartitionClusters(weld, clusters, pending);
            level_stats.PartitionMs += MillisecondsSince(partition_start);

            const auto lock_start = Clock::now();
            LockBoundary(weld, clusters, groups);
            level_stats.LockMs += MillisecondsSince(lock_start);

            const uint32_t group_base = uint32_t(build.Groups.size());
            scratch.assign(groups.size(), GroupScratch{});
            const auto run = [&](uint32_t i) { RunGroup(scratch[i], weld, clusters, groups[i], primitive, group_base + i); };
            if (serial) {
                for (uint32_t i = 0; i < groups.size(); ++i) run(i);
            } else {
                ParallelFor(uint32_t(groups.size()), run);
            }

            // Groups merge in partition order, so the DAG never depends on which group finished first.
            const auto merge_start = Clock::now();
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

                level_stats.Groups++;
                level_stats.Clusters += uint32_t(group_scratch.MemberLevel0.size());
                level_stats.Triangles += group_scratch.Triangles;
                level_stats.SingletonGroups += group_scratch.MemberLevel0.size() == 1u;
                level_stats.MeanRadius += group_scratch.RadiusSum;
                if (group_scratch.Stuck) {
                    level_stats.StuckClusters += uint32_t(group_scratch.MemberLevel0.size());
                    level_stats.StuckTriangles += group_scratch.Triangles;
                }
                level_stats.SimplifyMs += group_scratch.SimplifyMs;
                level_stats.ClusterizeMs += group_scratch.ClusterizeMs;
                level_stats.EmitMs += group_scratch.EmitMs;
            }
            level_stats.MergeMs += MillisecondsSince(merge_start);
            level_stats.LevelMs += MillisecondsSince(level_start);
            ++depth;
        }

        // A single remaining cluster forms a terminal group.
        if (pending.size() == 1u) {
            if (build.Stats.Levels.size() <= depth) build.Stats.Levels.emplace_back();
            auto &level_stats = build.Stats.Levels[depth];
            const auto level_start = Clock::now();
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
            level_stats.Groups++;
            level_stats.Clusters++;
            level_stats.Triangles += cluster.TriangleCount();
            level_stats.SingletonGroups++;
            level_stats.StuckClusters++;
            level_stats.StuckTriangles += cluster.TriangleCount();
            level_stats.MeanRadius += cluster.Sphere.Radius;
            level_stats.LevelMs += MillisecondsSince(level_start);
            ++depth;
        }

        primitive_range.ClusterCount = uint32_t(build.Clusters.size()) - primitive_range.FirstCluster;
        primitive_range.GroupCount = uint32_t(build.Groups.size()) - primitive_range.FirstGroup;
        build.PrimitiveRanges.push_back(primitive_range);
        build.LevelCount = std::max(build.LevelCount, depth);
    }

    for (auto &level : build.Stats.Levels) {
        level.MeanRadius = level.Clusters == 0u ? 0.f : level.MeanRadius / float(level.Clusters);
    }

    const auto hierarchy_start = Clock::now();
    BuildSpanTrees(build, mesh);
    build.Stats.HierarchyMs = MillisecondsSince(hierarchy_start);
    build.Stats.TotalMs = MillisecondsSince(total_start);
    return build;
}
