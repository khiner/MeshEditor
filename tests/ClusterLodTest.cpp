// Pins the cluster LOD DAG builder against the properties runtime selection depends on: the same
// bytes every run, errors that never shrink toward the coarse end, and a cut that covers each
// level-0 cluster exactly once at every threshold.

#include "render/ClusterLod.h"

#include "RunSuites.h"

#include "meshoptimizer.h"

#include <boost/ut.hpp>

#include <glm/geometric.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numbers>
#include <print>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

using namespace boost::ut;

namespace {
// Source geometry in the arrays the build reads.
struct Fixture {
    std::vector<vec3> Positions;
    std::vector<vec3> CornerNormals;
    std::vector<vec2> CornerUvs;
    std::vector<uint32_t> CornerVertices;
    std::vector<ClusterLodPrimitive> Primitives;
    std::vector<ClusterLodSourceCluster> Clusters;
    std::vector<uint32_t> ClusterVertices;
    std::vector<uint8_t> ClusterLocalTriangles;

    uint32_t TriangleCount() const { return uint32_t(CornerVertices.size() / 3u); }
};

// A UV sphere with its seam column duplicated, so the two copies share a position and disagree on u.
// Pole quads emit a single triangle each, which keeps every triangle non-degenerate.
void AppendSphere(Fixture &fixture, uint32_t rings, uint32_t segments, vec3 origin, float radius) {
    const uint32_t base_vertex = uint32_t(fixture.Positions.size());
    for (uint32_t r = 0; r <= rings; ++r) {
        const float theta = std::numbers::pi_v<float> * float(r) / float(rings);
        for (uint32_t s = 0; s <= segments; ++s) {
            const float phi = 2.f * std::numbers::pi_v<float> * float(s) / float(segments);
            fixture.Positions.push_back(origin + radius * vec3{std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)});
        }
    }
    const auto vertex = [&](uint32_t r, uint32_t s) { return base_vertex + r * (segments + 1u) + s; };
    const auto uv = [&](uint32_t r, uint32_t s) { return vec2{float(s) / float(segments), float(r) / float(rings)}; };
    const auto triangle = [&](uint32_t ra, uint32_t sa, uint32_t rb, uint32_t sb, uint32_t rc, uint32_t sc) {
        fixture.CornerVertices.insert(fixture.CornerVertices.end(), {vertex(ra, sa), vertex(rb, sb), vertex(rc, sc)});
        fixture.CornerUvs.insert(fixture.CornerUvs.end(), {uv(ra, sa), uv(rb, sb), uv(rc, sc)});
    };
    for (uint32_t r = 0; r < rings; ++r) {
        for (uint32_t s = 0; s < segments; ++s) {
            if (r + 1u != rings) triangle(r, s, r + 1u, s + 1u, r + 1u, s);
            if (r != 0u) triangle(r, s, r, s + 1u, r + 1u, s + 1u);
        }
    }
}

// A flat grid whose middle column is duplicated with a one-unit jump in u, which is the sharp seam
// permissive simplification must not collapse across.
void AppendSeamGrid(Fixture &fixture, uint32_t cells) {
    const uint32_t base_vertex = uint32_t(fixture.Positions.size());
    const uint32_t seam = cells / 2u;
    const uint32_t columns = cells + 2u; // the seam column appears twice
    const auto column_x = [&](uint32_t column) { return column <= seam ? column : column - 1u; };
    for (uint32_t y = 0; y <= cells; ++y) {
        for (uint32_t column = 0; column <= columns - 1u; ++column) {
            fixture.Positions.push_back(vec3{float(column_x(column)) / float(cells), 0.f, float(y) / float(cells)});
        }
    }
    const auto vertex = [&](uint32_t x, uint32_t y) { return base_vertex + y * columns + x; };
    const auto uv = [&](uint32_t x, uint32_t y) {
        return vec2{float(column_x(x)) / float(cells) + (x > seam ? 1.f : 0.f), float(y) / float(cells)};
    };
    const auto triangle = [&](uint32_t xa, uint32_t ya, uint32_t xb, uint32_t yb, uint32_t xc, uint32_t yc) {
        fixture.CornerVertices.insert(fixture.CornerVertices.end(), {vertex(xa, ya), vertex(xb, yb), vertex(xc, yc)});
        fixture.CornerUvs.insert(fixture.CornerUvs.end(), {uv(xa, ya), uv(xb, yb), uv(xc, yc)});
        fixture.CornerNormals.insert(fixture.CornerNormals.end(), {vec3{0, 1, 0}, vec3{0, 1, 0}, vec3{0, 1, 0}});
    };
    for (uint32_t y = 0; y < cells; ++y) {
        for (uint32_t x = 0; x < cells; ++x) {
            // The column left of the seam pairs with the second copy, so no triangle spans the jump.
            const uint32_t left = x < seam ? x : x + 1u;
            const uint32_t right = left + 1u;
            triangle(left, y, right, y, right, y + 1u);
            triangle(left, y, right, y + 1u, left, y + 1u);
        }
    }
}

void AppendLevel0Clusters(Fixture &fixture, uint32_t first_triangle, uint32_t triangle_count) {
    const auto indices = std::span{fixture.CornerVertices}.subspan(size_t(first_triangle) * 3u, size_t(triangle_count) * 3u);
    const auto bound = meshopt_buildMeshletsBound(indices.size(), ClusterLodMaxVertices, ClusterLodMaxTriangles);
    std::vector<meshopt_Meshlet> built(bound);
    std::vector<uint32_t> local_vertices(bound * ClusterLodMaxVertices);
    std::vector<uint8_t> local_triangles(bound * ClusterLodMaxTriangles * 3u);
    built.resize(meshopt_buildMeshlets(
        built.data(), local_vertices.data(), local_triangles.data(), indices.data(), indices.size(),
        &fixture.Positions.front().x, fixture.Positions.size(), sizeof(vec3),
        ClusterLodMaxVertices, ClusterLodMaxTriangles, 0.5f
    ));
    std::vector<uint32_t> representative(fixture.Positions.size(), ClusterLodInvalid);
    for (uint32_t corner = 0; corner < indices.size(); ++corner) {
        auto &first = representative[indices[corner]];
        if (first == ClusterLodInvalid) first = corner;
    }

    for (const auto &meshlet : built) {
        const uint32_t first_vertex = uint32_t(fixture.ClusterVertices.size());
        for (uint32_t v = 0; v < meshlet.vertex_count; ++v) {
            const uint32_t vertex = local_vertices[meshlet.vertex_offset + v];
            fixture.ClusterVertices.push_back(representative[vertex]);
        }
        const uint32_t first_local_triangle = uint32_t(fixture.ClusterLocalTriangles.size());
        fixture.ClusterLocalTriangles.insert(
            fixture.ClusterLocalTriangles.end(),
            local_triangles.begin() + meshlet.triangle_offset,
            local_triangles.begin() + meshlet.triangle_offset + size_t(meshlet.triangle_count) * 3u
        );
        const auto bounds = meshopt_computeMeshletBounds(
            &local_vertices[meshlet.vertex_offset], &local_triangles[meshlet.triangle_offset], meshlet.triangle_count,
            &fixture.Positions.front().x, fixture.Positions.size(), sizeof(vec3)
        );
        fixture.Clusters.push_back(ClusterLodSourceCluster{
            .FirstVertex = first_vertex,
            .VertexCount = meshlet.vertex_count,
            .FirstLocalTriangle = first_local_triangle,
            .TriangleCount = meshlet.triangle_count,
            .Center = {bounds.center[0], bounds.center[1], bounds.center[2]},
            .Radius = bounds.radius,
            .ConeCullSafe = true,
        });
    }
}

void CloseTrianglePrimitive(Fixture &fixture, uint32_t first_triangle) {
    const uint32_t triangle_count = fixture.TriangleCount() - first_triangle;
    const uint32_t first_cluster = uint32_t(fixture.Clusters.size());
    AppendLevel0Clusters(fixture, first_triangle, triangle_count);
    fixture.Primitives.push_back(ClusterLodPrimitive{
        .FirstTriangle = first_triangle,
        .TriangleCount = triangle_count,
        .FirstCluster = first_cluster,
        .ClusterCount = uint32_t(fixture.Clusters.size()) - first_cluster,
    });
}

Fixture SphereFixture(uint32_t rings, uint32_t segments) {
    Fixture fixture;
    AppendSphere(fixture, rings, segments, vec3{0}, 1.f);
    CloseTrianglePrimitive(fixture, 0u);
    return fixture;
}

Fixture SeamGridFixture(uint32_t cells) {
    Fixture fixture;
    AppendSeamGrid(fixture, cells);
    CloseTrianglePrimitive(fixture, 0u);
    return fixture;
}

Fixture TriangleSoupFixture(uint32_t triangle_count) {
    Fixture fixture;
    fixture.Positions.reserve(size_t(triangle_count) * 3u);
    fixture.CornerVertices.reserve(size_t(triangle_count) * 3u);
    for (uint32_t triangle = 0; triangle < triangle_count; ++triangle) {
        const float x = float(triangle % 300u);
        const float y = float(triangle / 300u);
        const uint32_t first = uint32_t(fixture.Positions.size());
        fixture.Positions.insert(fixture.Positions.end(), {{x, y, 0.f}, {x + 0.4f, y, 0.f}, {x, y + 0.4f, 0.f}});
        fixture.CornerVertices.insert(fixture.CornerVertices.end(), {first, first + 1u, first + 2u});
    }
    CloseTrianglePrimitive(fixture, 0u);
    return fixture;
}

Fixture TwoPrimitiveFixture(uint32_t rings, uint32_t segments) {
    Fixture fixture;
    AppendSphere(fixture, rings, segments, vec3{0}, 1.f);
    CloseTrianglePrimitive(fixture, 0u);
    const uint32_t second = fixture.TriangleCount();
    AppendSphere(fixture, rings, segments, vec3{4, 0, 0}, 0.5f);
    CloseTrianglePrimitive(fixture, second);
    return fixture;
}

ClusterLodMesh MeshOf(const Fixture &fixture) {
    ClusterLodMesh mesh;
    mesh.CornerVertices = fixture.CornerVertices;
    mesh.Positions = &fixture.Positions.front().x;
    mesh.PositionStride = sizeof(vec3);
    mesh.CornerNormals = fixture.CornerNormals;
    mesh.Weld.CornerUvs[0] = fixture.CornerUvs;
    mesh.Primitives = fixture.Primitives;
    mesh.Clusters = fixture.Clusters;
    mesh.SourceVertexCorners = fixture.ClusterVertices;
    mesh.SourceLocalTriangles = fixture.ClusterLocalTriangles;
    return mesh;
}

template<typename T> bool SameBytes(const std::vector<T> &a, const std::vector<T> &b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

bool SameBuild(const ClusterLodBuild &a, const ClusterLodBuild &b) {
    return SameBytes(a.Clusters, b.Clusters) && SameBytes(a.VertexCorners, b.VertexCorners) &&
        SameBytes(a.LocalTriangles, b.LocalTriangles) && SameBytes(a.Groups, b.Groups) &&
        SameBytes(a.GroupClusters, b.GroupClusters) && SameBytes(a.Nodes, b.Nodes) &&
        SameBytes(a.Level0Groups, b.Level0Groups) && SameBytes(a.PrimitiveRanges, b.PrimitiveRanges) &&
        a.LevelCount == b.LevelCount && a.NodeDepth == b.NodeDepth;
}

// The group whose simplification replaces a cluster, and the group the cluster came out of.
uint32_t ParentGroup(const ClusterLodBuild &build, uint32_t cluster) {
    return cluster < build.Level0Count() ? build.Level0Groups[cluster] : build.Clusters[cluster - build.Level0Count()].GroupIndex;
}
uint32_t ChildGroup(const ClusterLodBuild &build, uint32_t cluster) {
    return cluster < build.Level0Count() ? ClusterLodInvalid : build.Clusters[cluster - build.Level0Count()].RefinedGroup;
}

// The screen-space error a group projects to, which must grow toward the coarse end for a cut to hold.
float ProjectedError(const ClusterLodGroup &group, vec3 camera, float proj, float znear) {
    const float distance = glm::length(group.Center - camera) - group.Radius;
    return group.Error / std::max(distance, znear) * (proj * 0.5f);
}

bool Selected(const ClusterLodBuild &build, uint32_t cluster, vec3 camera, float proj, float znear, float threshold) {
    if (ProjectedError(build.Groups[ParentGroup(build, cluster)], camera, proj, znear) <= threshold) return false;
    const uint32_t child = ChildGroup(build, cluster);
    return child == ClusterLodInvalid || ProjectedError(build.Groups[child], camera, proj, znear) <= threshold;
}

// One render record's own sphere and the group the cut reads for it, in the order the render arena
// lays a mesh's primitives out.
struct SpanRecord {
    vec3 Center;
    float Radius;
    uint32_t Group;
};

std::vector<SpanRecord> SpanRecords(const ClusterLodBuild &build, const ClusterLodMesh &mesh) {
    std::vector<SpanRecord> records;
    for (uint32_t p = 0; p < mesh.Primitives.size(); ++p) {
        const auto &source = mesh.Primitives[p];
        const auto &range = build.PrimitiveRanges[p];
        for (uint32_t k = 0; k < source.ClusterCount; ++k) {
            const auto &cluster = mesh.Clusters[source.FirstCluster + k];
            records.push_back({cluster.Center, cluster.Radius, build.Level0Groups[source.FirstCluster + k]});
        }
        for (uint32_t c = 0; c < range.ClusterCount; ++c) {
            const auto &cluster = build.Clusters[range.FirstCluster + c];
            records.push_back({cluster.Center, cluster.Radius, cluster.GroupIndex});
        }
    }
    return records;
}

bool Held(vec3 center, float radius, const LodNode &node) {
    return glm::length(center - node.Center) + radius <= node.Radius + 1e-4f * std::max(node.Radius, 1.f);
}

std::vector<std::vector<uint32_t>> ProducedBy(const ClusterLodBuild &build) {
    std::vector<std::vector<uint32_t>> produced(build.Groups.size());
    for (uint32_t i = 0; i < build.Clusters.size(); ++i) {
        const uint32_t refined = build.Clusters[i].RefinedGroup;
        if (refined != ClusterLodInvalid) produced[refined].push_back(build.Level0Count() + i);
    }
    return produced;
}

void ReportLevels(const ClusterLodBuild &build, std::string_view title) {
    std::println("{}: {} levels, {} coarse clusters, {} groups, {} nodes, build {:.1f} ms (weld {:.1f}, level 0 {:.1f}, hierarchy {:.1f})",
                 title, build.LevelCount, build.Clusters.size(), build.Groups.size(), build.Nodes.size(),
                 build.Stats.TotalMs, build.Stats.WeldMs, build.Stats.Level0Ms, build.Stats.HierarchyMs);
    for (uint32_t level = 0; level < build.Stats.Levels.size(); ++level) {
        const auto &stats = build.Stats.Levels[level];
        std::println("  level {}: {} groups, {} clusters, {} triangles, {} stuck clusters ({} triangles), {} singletons, mean radius {:.4f}",
                     level, stats.Groups, stats.Clusters, stats.Triangles, stats.StuckClusters, stats.StuckTriangles, stats.SingletonGroups, stats.MeanRadius);
        std::println("           wall {:.1f} ms (partition {:.1f}, lock {:.1f}, merge {:.1f}), group cpu: simplify {:.1f}, clusterize {:.1f}, emit {:.1f}",
                     stats.LevelMs, stats.PartitionMs, stats.LockMs, stats.MergeMs, stats.SimplifyMs, stats.ClusterizeMs, stats.EmitMs);
    }
}
} // namespace

int main() {
    const auto small_sphere = SphereFixture(32u, 64u);
    const auto medium_sphere = SphereFixture(128u, 256u);
    const auto seam_grid = SeamGridFixture(200u);
    const auto two_primitives = TwoPrimitiveFixture(24u, 48u);

    const auto small_build = BuildClusterLod(MeshOf(small_sphere));
    const auto grid_build = BuildClusterLod(MeshOf(seam_grid));
    const auto two_build = BuildClusterLod(MeshOf(two_primitives));
    const std::array<std::pair<const ClusterLodBuild *, const Fixture *>, 3> builds{{
        {&small_build, &small_sphere}, {&grid_build, &seam_grid}, {&two_build, &two_primitives},
    }};

    "a mesh with no clusters builds nothing"_test = [] {
        expect(BuildClusterLod(ClusterLodMesh{}).Groups.empty());
    };

    "the same input builds the same bytes twice"_test = [&] {
        const auto first = BuildClusterLod(MeshOf(medium_sphere));
        const auto second = BuildClusterLod(MeshOf(medium_sphere));
        expect(SameBuild(first, second));
        expect(first.Groups.size() > 1_ul);
    };

    "running a level's groups one at a time builds the same bytes"_test = [&] {
        const auto parallel = BuildClusterLod(MeshOf(medium_sphere));
        const auto serial = BuildClusterLod(MeshOf(medium_sphere), /*serial=*/true);
        expect(SameBuild(parallel, serial));
    };

    "the parallel position remap matches the reference bytes eight times"_test = [] {
        const auto remap_mesh = TriangleSoupFixture(90'000u);
        const auto reference = BuildClusterLod(MeshOf(remap_mesh), /*serial=*/true);
        for (uint32_t run = 0; run < 8u; ++run) expect(SameBuild(BuildClusterLod(MeshOf(remap_mesh)), reference));
    };

    "the source-vertex weld matches the general key path"_test = [] {
        auto fixture = SphereFixture(32u, 64u);
        fixture.CornerUvs.clear();
        const auto direct = BuildClusterLod(MeshOf(fixture));

        auto generic_mesh = MeshOf(fixture);
        std::vector<uint32_t> corner_classes(
            fixture.CornerVertices.size(),
            uint32_t(CornerClass::Vertex) << uint32_t(CornerClassEncoding::TagShift)
        );
        generic_mesh.Weld.CornerClasses = corner_classes;
        const auto generic = BuildClusterLod(generic_mesh);
        expect(SameBuild(direct, generic));
    };

    "level-zero records carry their reconstructed bounds"_test = [&] {
        for (const auto &entry : builds) {
            const auto *fixture = entry.second;
            for (const auto &primitive : fixture->Primitives) {
                for (uint32_t i = 0; i < primitive.ClusterCount; ++i) {
                    const auto &cluster = fixture->Clusters[primitive.FirstCluster + i];
                    const auto local_triangles = std::span{fixture->ClusterLocalTriangles}.subspan(
                        cluster.FirstLocalTriangle, size_t(cluster.TriangleCount) * 3u
                    );
                    std::vector<uint32_t> indices;
                    indices.reserve(local_triangles.size());
                    for (const uint8_t local : local_triangles) {
                        const uint32_t corner = fixture->ClusterVertices[cluster.FirstVertex + local];
                        indices.push_back(fixture->CornerVertices[primitive.FirstTriangle * 3u + corner]);
                    }
                    const auto bounds = meshopt_computeClusterBounds(
                        indices.data(), indices.size(), &fixture->Positions.front().x, fixture->Positions.size(), sizeof(vec3)
                    );
                    expect(cluster.Center.x == bounds.center[0] && cluster.Center.y == bounds.center[1] && cluster.Center.z == bounds.center[2] && cluster.Radius == bounds.radius);
                }
            }
        }
    };

    "every level-0 cluster lands in exactly one group"_test = [&] {
        for (const auto &[build, fixture] : builds) {
            std::vector<uint32_t> counts(build->Groups.size(), 0u);
            for (const auto group : build->Level0Groups) {
                expect(group < build->Groups.size());
                counts[group]++;
            }
            uint32_t members = 0;
            for (const auto &group : build->Groups) members += group.ClusterCount;
            expect(members == build->GroupClusters.size());
            expect(build->GroupClusters.size() == build->Level0Count() + build->Clusters.size());
            std::vector<uint8_t> seen(build->GroupClusters.size(), 0u);
            for (const auto cluster : build->GroupClusters) {
                expect(cluster < seen.size());
                expect(seen[cluster] == 0u);
                seen[cluster] = 1u;
            }
        }
    };

    "errors never shrink toward the coarse end"_test = [&] {
        for (const auto &[build, fixture] : builds) {
            for (uint32_t i = 0; i < build->Clusters.size(); ++i) {
                const auto &cluster = build->Clusters[i];
                expect(cluster.RefinedGroup < build->Groups.size());
                expect(build->Groups[cluster.RefinedGroup].Error <= build->Groups[cluster.GroupIndex].Error);
            }
            for (uint32_t i = 0; i < build->Level0Count(); ++i) {
                expect(build->Groups[build->Level0Groups[i]].Error > 0.f);
            }
        }
    };

    "a group's sphere holds the spheres it merged"_test = [&] {
        for (const auto &[build, fixture] : builds) {
            for (uint32_t g = 0; g < build->Groups.size(); ++g) {
                const auto &group = build->Groups[g];
                for (uint32_t i = 0; i < group.ClusterCount; ++i) {
                    const uint32_t id = build->GroupClusters[group.FirstCluster + i];
                    if (id < build->Level0Count()) continue;
                    const auto &child = build->Groups[build->Clusters[id - build->Level0Count()].RefinedGroup];
                    const float slack = glm::length(child.Center - group.Center) + child.Radius - group.Radius;
                    expect(slack <= 1e-4f * std::max(group.Radius, 1.f));
                }
            }
        }
    };

    "a span node holds every sphere and error its run covers"_test = [&] {
        for (const auto &[build, fixture] : builds) {
            const auto records = SpanRecords(*build, MeshOf(*fixture));
            for (const auto &node : build->Nodes) {
                if (std::isinf(node.Error)) continue;
                for (uint32_t i = 0; i < node.MeshletCount; ++i) {
                    const auto &record = records[node.FirstMeshlet + i];
                    const auto &group = build->Groups[record.Group];
                    expect(group.Error <= node.Error);
                    expect(Held(record.Center, record.Radius, node));
                    expect(Held(group.Center, group.Radius, node));
                }
            }
        }
    };

    "a primitive's span tree tiles its records in order"_test = [&] {
        for (const auto &[build, fixture] : builds) {
            const auto mesh = MeshOf(*fixture);
            uint32_t first_record = 0;
            for (uint32_t p = 0; p < build->PrimitiveRanges.size(); ++p) {
                const auto &range = build->PrimitiveRanges[p];
                const uint32_t count = mesh.Primitives[p].ClusterCount + range.ClusterCount;
                uint32_t next = first_record;
                const auto walk = [&](auto &&self, uint32_t index) -> void {
                    const auto &node = build->Nodes[index];
                    if (node.ChildCount == 0u) {
                        expect(node.FirstMeshlet == next);
                        next += node.MeshletCount;
                        return;
                    }
                    for (uint32_t c = 0; c < node.ChildCount; ++c) self(self, node.ChildOffset + c);
                };
                walk(walk, range.RootNode);
                expect(next == first_record + count);
                const auto &finest = build->Nodes[range.FinestNode];
                expect(finest.FirstMeshlet == first_record);
                expect(finest.MeshletCount == mesh.Primitives[p].ClusterCount);
                expect(std::isinf(finest.Error));
                first_record += count;
            }
        }
    };

    "a cut at threshold zero selects exactly the level-0 clusters"_test = [&] {
        for (const auto &[build, fixture] : builds) {
            const uint32_t total = build->Level0Count() + uint32_t(build->Clusters.size());
            for (const vec3 camera : {vec3{0, 0, 5}, vec3{20, 3, -7}, vec3{0.2f, 0.1f, 0.3f}}) {
                uint32_t selected = 0;
                for (uint32_t id = 0; id < total; ++id) {
                    const bool keep = Selected(*build, id, camera, 1.7320508f, 1e-2f, 0.f);
                    expect(keep == (id < build->Level0Count()));
                    selected += keep;
                }
                expect(selected == build->Level0Count());
            }
        }
    };

    "every refinement chain has exactly one cluster in the cut"_test = [&] {
        for (const auto &[build, fixture] : builds) {
            const auto produced = ProducedBy(*build);
            for (const vec3 camera : {vec3{0, 0, 3}, vec3{0, 0, 30}, vec3{0, 0, 300}}) {
                for (const float threshold : {1e-4f, 1e-3f, 1e-2f, 1e-1f}) {
                    for (uint32_t start = 0; start < build->Level0Count(); ++start) {
                        // A chain follows one level-0 cluster's geometry up the DAG, taking the
                        // first cluster each group simplified it into.
                        uint32_t chain = start, selected = 0, length = 0;
                        while (true) {
                            selected += Selected(*build, chain, camera, 1.7320508f, 1e-2f, threshold);
                            const auto &next = produced[ParentGroup(*build, chain)];
                            if (next.empty()) break;
                            chain = next.front();
                            expect(++length < 64u);
                        }
                        expect(selected == 1u);
                    }
                }
            }
        }
    };

    "every emitted cluster fits the mesh shader contract"_test = [&] {
        for (const auto &[build, fixture] : builds) {
            for (const auto &cluster : build->Clusters) {
                expect(cluster.VertexCount <= ClusterLodMaxVertices);
                expect(cluster.TriangleCount <= ClusterLodMaxTriangles);
                expect(cluster.VertexCount > 0u && cluster.TriangleCount > 0u);
                expect(cluster.VertexOffset + cluster.VertexCount <= build->VertexCorners.size());
                expect(size_t(cluster.LocalTriangleOffset) + size_t(cluster.TriangleCount) * 3u <= build->LocalTriangles.size());
                for (uint32_t i = 0; i < cluster.TriangleCount * 3u; ++i) {
                    expect(build->LocalTriangles[cluster.LocalTriangleOffset + i] < cluster.VertexCount);
                }
                expect(cluster.Primitive == build->Groups[cluster.GroupIndex].Primitive);
                expect(cluster.Primitive == build->Groups[cluster.RefinedGroup].Primitive);
            }
        }
    };

    "every emitted vertex names a corner of its own primitive"_test = [&] {
        for (const auto &[build, fixture] : builds) {
            for (const auto &cluster : build->Clusters) {
                const auto &primitive = fixture->Primitives[cluster.Primitive];
                for (uint32_t i = 0; i < cluster.VertexCount; ++i) {
                    expect(build->VertexCorners[cluster.VertexOffset + i] < primitive.TriangleCount * 3u);
                }
            }
        }
    };

    "a cluster never spans two primitives"_test = [&] {
        expect(two_build.PrimitiveRanges.size() == 2_ul);
        std::vector<uint32_t> level0_primitive(two_build.Level0Count(), ClusterLodInvalid);
        for (uint32_t p = 0; p < two_primitives.Primitives.size(); ++p) {
            const auto &primitive = two_primitives.Primitives[p];
            for (uint32_t i = 0; i < primitive.ClusterCount; ++i) level0_primitive[primitive.FirstCluster + i] = p;
        }
        for (uint32_t p = 0; p < two_build.PrimitiveRanges.size(); ++p) {
            const auto &range = two_build.PrimitiveRanges[p];
            expect(range.ClusterCount > 0u && range.GroupCount > 0u);
            for (uint32_t i = 0; i < range.ClusterCount; ++i) {
                expect(two_build.Clusters[range.FirstCluster + i].Primitive == p);
            }
            for (uint32_t i = 0; i < range.GroupCount; ++i) {
                const auto &group = two_build.Groups[range.FirstGroup + i];
                expect(group.Primitive == p);
                for (uint32_t c = 0; c < group.ClusterCount; ++c) {
                    const uint32_t id = two_build.GroupClusters[group.FirstCluster + c];
                    const uint32_t primitive = id < two_build.Level0Count()
                        ? level0_primitive[id]
                        : two_build.Clusters[id - two_build.Level0Count()].Primitive;
                    expect(primitive == p);
                }
            }
        }
    };

    "a coarse cluster over flat geometry keeps a culling cone"_test = [&] {
        uint32_t coned = 0;
        for (const auto &cluster : grid_build.Clusters) coned += (cluster.ConeAxisCutoff >> 24u) != 127u;
        expect(coned > 0u);
    };

    "the UV seam survives to the coarsest level"_test = [&] {
        const auto &top = grid_build.PrimitiveRanges.front();
        const uint32_t last_group = top.FirstGroup + top.GroupCount - 1u;
        std::vector<vec3> positions;
        std::vector<vec2> uvs;
        for (const auto &cluster : grid_build.Clusters) {
            if (cluster.GroupIndex != last_group) continue;
            for (uint32_t i = 0; i < cluster.VertexCount; ++i) {
                const uint32_t corner = grid_build.VertexCorners[cluster.VertexOffset + i];
                positions.push_back(seam_grid.Positions[seam_grid.CornerVertices[corner]]);
                uvs.push_back(seam_grid.CornerUvs[corner]);
            }
        }
        bool split = false;
        for (size_t i = 0; i < positions.size() && !split; ++i) {
            for (size_t j = i + 1; j < positions.size() && !split; ++j) {
                split = positions[i] == positions[j] && uvs[i] != uvs[j];
            }
        }
        expect(split);
    };

    "the build cost of a million triangles is reported"_test = [] {
        const auto fixture = SphereFixture(512u, 1024u);
        std::println("large sphere: {} triangles, {} level-0 clusters", fixture.TriangleCount(), fixture.Clusters.size());
        const auto build = BuildClusterLod(MeshOf(fixture));
        ReportLevels(build, "large sphere");
        expect(build.LevelCount > 4u);
    };

    ReportLevels(small_build, "small sphere");
    ReportLevels(grid_build, "seam grid");
    return RunSuites();
}
