#include "render/ClusterLod.h"
#include "RunSuites.h"
#include "meshoptimizer.h"

#include <bit>
#include <cmath>
#include <numbers>
#include <set>

using namespace boost::ut;

namespace {
struct Fixture {
    std::vector<vec3> Positions;
    std::vector<vec3> VertexNormals;
    std::vector<vec2> CornerUvs;
    std::vector<uint32_t> AttributeBlocks, HalfedgeFaces;
    std::vector<uvec3> Triangles;
    std::vector<uint32_t> CornerVertices;
    std::vector<ClusterLodPrimitive> Primitives;
    std::vector<ClusterLodSourceCluster> Clusters;
    std::vector<uint32_t> ClusterVertices;
    std::vector<uint8_t> ClusterLocalTriangles;

    uint32_t TriangleCount() const { return uint32_t(CornerVertices.size() / 3u); }
};

void AppendSphere(Fixture &fixture, uint32_t rings, uint32_t segments, vec3 origin, float radius) {
    const uint32_t base_vertex = uint32_t(fixture.Positions.size());
    for (uint32_t r = 0; r <= rings; ++r) {
        const float theta = std::numbers::pi_v<float> * float(r) / float(rings);
        for (uint32_t s = 0; s <= segments; ++s) {
            const float phi = 2.f * std::numbers::pi_v<float> * float(s) / float(segments);
            const vec3 normal{std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)};
            fixture.Positions.push_back(origin + radius * normal);
            fixture.VertexNormals.push_back(normal);
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

void AppendSeamGrid(Fixture &fixture, uint32_t cells) {
    const uint32_t base_vertex = uint32_t(fixture.Positions.size());
    const uint32_t seam = cells / 2u;
    const uint32_t columns = cells + 2u;
    const auto column_x = [&](uint32_t column) { return column <= seam ? column : column - 1u; };
    for (uint32_t y = 0; y <= cells; ++y) {
        for (uint32_t column = 0; column <= columns - 1u; ++column) {
            fixture.Positions.push_back(vec3{float(column_x(column)) / float(cells), 0.f, float(y) / float(cells)});
            fixture.VertexNormals.push_back({0, 1, 0});
        }
    }
    const auto vertex = [&](uint32_t x, uint32_t y) { return base_vertex + y * columns + x; };
    const auto uv = [&](uint32_t x, uint32_t y) {
        return vec2{float(column_x(x)) / float(cells) + (x > seam ? 1.f : 0.f), float(y) / float(cells)};
    };
    const auto triangle = [&](uint32_t xa, uint32_t ya, uint32_t xb, uint32_t yb, uint32_t xc, uint32_t yc) {
        fixture.CornerVertices.insert(fixture.CornerVertices.end(), {vertex(xa, ya), vertex(xb, yb), vertex(xc, yc)});
        fixture.CornerUvs.insert(fixture.CornerUvs.end(), {uv(xa, ya), uv(xb, yb), uv(xc, yc)});
    };
    for (uint32_t y = 0; y < cells; ++y) {
        for (uint32_t x = 0; x < cells; ++x) {
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
            fixture.ClusterVertices.push_back(first_triangle * 3u + representative[vertex]);
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
    while (fixture.Triangles.size() < fixture.TriangleCount()) {
        const auto face = uint32_t(fixture.Triangles.size());
        fixture.Triangles.push_back({face * 3u, face * 3u + 1u, face * 3u + 2u});
        fixture.HalfedgeFaces.insert(fixture.HalfedgeFaces.end(), 3u, face);
    }
    while (fixture.AttributeBlocks.size() * MeshElementBlockSize < fixture.CornerVertices.size())
        fixture.AttributeBlocks.push_back(uint32_t(fixture.AttributeBlocks.size()) + 1u);
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

TriangleCorners TriangleRefs(const auto &triangles) { return {ElementView<uvec3>{std::span{triangles}}}; }

ClusterLodMesh MeshOf(const Fixture &fixture) {
    return {
        .CornerVertices = fixture.CornerVertices,
        .Positions = &fixture.Positions.front().x,
        .PositionStride = sizeof(vec3),
        .DenseVertices = {0u, uint32_t(fixture.Positions.size())},
        .Normals = {.CornerVertices = fixture.CornerVertices, .VertexNormals = fixture.VertexNormals},
        .Weld = {.CornerClassMode = uint32_t(CornerClassMode::UniformVertex), .TriangleFaces = {TriangleRefs(fixture.Triangles), fixture.HalfedgeFaces}, .CornerUvs = {CornerAttributeView<vec2>{{fixture.AttributeBlocks, fixture.CornerUvs}, TriangleRefs(fixture.Triangles)}}},
        .Primitives = fixture.Primitives,
        .Clusters = fixture.Clusters,
        .SourceVertexCorners = fixture.ClusterVertices,
        .SourceLocalTriangles = fixture.ClusterLocalTriangles,
    };
}

// Compare every published field without depending on struct padding.
std::vector<uint32_t> LodWords(const ClusterLodBuild &build) {
    std::vector<uint32_t> words;
    const auto put = [&](auto... values) { (words.push_back(std::bit_cast<uint32_t>(values)), ...); };
    const auto list = [&](const auto &values) { put(uint32_t(values.size())); for (const auto value:values) put(uint32_t(value)); };
    put(build.LevelCount, build.NodeDepth, uint32_t(build.Clusters.size()));
    for (const auto &c : build.Clusters) put(c.VertexOffset, c.VertexCount, c.LocalTriangleOffset, c.TriangleCount, c.Primitive, c.ConeAxisCutoff, c.Center.x, c.Center.y, c.Center.z, c.Radius, c.GroupIndex, c.RefinedGroup);
    list(build.VertexCorners);
    list(build.LocalTriangles);
    put(uint32_t(build.Groups.size()));
    for (const auto &g : build.Groups) put(g.Center.x, g.Center.y, g.Center.z, g.Radius, g.Error, g.FirstCluster, g.ClusterCount, g.Primitive);
    list(build.GroupClusters);
    list(build.Level0Groups);
    put(uint32_t(build.Nodes.size()));
    for (const auto &n : build.Nodes) put(n.Center.x, n.Center.y, n.Center.z, n.Radius, n.Error, n.FirstMeshlet, n.MeshletCount, n.ChildOffset, n.ChildCount, n.MeshletRoot);
    put(uint32_t(build.PrimitiveRanges.size()));
    for (const auto &p : build.PrimitiveRanges) put(p.FirstCluster, p.ClusterCount, p.FirstGroup, p.GroupCount, p.RootNode, p.FinestNode, p.SimplifyScale);
    return words;
}

} // namespace

int main() {
    "serial and parallel LOD builds preserve identical geometry bounds and errors"_test = [] {
        for (const bool seam : {false, true}) {
            const auto fixture = seam ? SeamGridFixture(48u) : SphereFixture(48u, 96u);
            const auto mesh = MeshOf(fixture);
            const auto reference = BuildClusterLod(mesh, true);
            expect(reference.LevelCount > 2u);
            for (uint32_t repeat = 0u; repeat < 3u; ++repeat) expect(LodWords(BuildClusterLod(mesh)) == LodWords(reference));
        }
    };
    "LOD repair handles members joining across levels deterministically"_test = [] {
        const auto fixture = SeamGridFixture(48u);
        auto mesh = MeshOf(fixture);
        std::vector<uvec3> triangles;
        std::vector<uint32_t> levels;
        for (uint32_t i = 0u; i < fixture.Clusters.size(); ++i) {
            const auto &c = fixture.Clusters[i];
            levels.push_back((i % 3u) * 2u);
            for (uint32_t t = 0u; t < c.TriangleCount; ++t) {
                uvec3 triangle;
                for (uint32_t k = 0u; k < 3u; ++k) triangle[k] = fixture.ClusterVertices[c.FirstVertex + fixture.ClusterLocalTriangles[c.FirstLocalTriangle + 3u * t + k]];
                triangles.push_back(triangle);
            }
        }
        const auto refs = TriangleRefs(triangles);
        mesh.CornerVertices = {refs, fixture.CornerVertices};
        mesh.Weld.TriangleFaces.Corners = refs;
        mesh.Weld.CornerUvs[0].Corners = refs;
        const auto scale = meshopt_simplifyScale(&fixture.Positions.front().x, fixture.Positions.size(), sizeof(vec3));
        const auto reference = RebuildClusterLod(mesh, levels, scale);
        expect(reference.LevelCount >= 5u);
        expect(reference.Level0Count() == levels.size());
        for (const auto group : reference.Level0Groups) expect(group < reference.Groups.size());
        for (uint32_t repeat = 0u; repeat < 3u; ++repeat) expect(LodWords(RebuildClusterLod(mesh, levels, scale)) == LodWords(reference));
    };
    "simplification retains an original-geometry cut"_test = [] {
        const auto fixture = SphereFixture(32u, 64u);
        const auto build = BuildClusterLod(MeshOf(fixture));
        expect(!build.Clusters.empty());
        expect(build.LevelCount > 1u);
        expect(build.Level0Count() == fixture.Clusters.size());
        expect(build.PrimitiveRanges.size() == 1u);
        if (build.PrimitiveRanges.empty()) return;
        const auto &finest = build.Nodes.at(build.PrimitiveRanges.front().FinestNode);
        expect(finest.FirstMeshlet == 0u);
        expect(finest.MeshletCount == fixture.Clusters.size());
    };

    "coarse geometry retains both sides of a UV seam"_test = [] {
        const auto fixture = SeamGridFixture(32u);
        const auto build = BuildClusterLod(MeshOf(fixture));
        expect(!build.Groups.empty());
        if (build.Groups.empty()) return;
        const auto terminal = uint32_t(build.Groups.size() - 1u);
        std::set<float> left, right;
        for (const auto &cluster : build.Clusters) {
            if (cluster.GroupIndex != terminal) continue;
            for (const auto corner : std::span{build.VertexCorners}.subspan(cluster.VertexOffset, cluster.VertexCount)) {
                const auto position = fixture.Positions[fixture.CornerVertices[corner]];
                if (position.x != 0.5f) continue;
                (fixture.CornerUvs[corner].x < 1.f ? left : right).insert(position.z);
            }
        }
        expect(!left.empty());
        expect(left == right);
    };
    return RunSuites();
}
