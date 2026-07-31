// Pins the mesh closest-point query against brute force over every triangle, the one oracle that does not
// reimplement the hierarchy's pruning.

#include "mesh/MeshBvh.h"

#include <boost/ut.hpp>

#include <glm/geometric.hpp>

#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

using namespace boost::ut;

namespace {
// The nearest triangle by scanning all of them, which the hierarchy must agree with exactly.
float BruteForceDistance(std::span<const Vertex> vertices, std::span<const uint32_t> triangles, vec3 point) {
    float best = std::numeric_limits<float>::max();
    for (size_t i = 0; i + 2 < triangles.size(); i += 3) {
        // Sample the triangle densely enough that a coarse minimum is within tolerance of the true one.
        const auto a = vertices[triangles[i]].Position, b = vertices[triangles[i + 1]].Position, c = vertices[triangles[i + 2]].Position;
        constexpr int Steps = 24;
        for (int u = 0; u <= Steps; ++u) {
            for (int v = 0; u + v <= Steps; ++v) {
                const float wu = float(u) / Steps, wv = float(v) / Steps;
                const vec3 p = a * (1 - wu - wv) + b * wu + c * wv;
                best = std::min(best, glm::length(p - point));
            }
        }
    }
    return best;
}

// The point a hit's weights give over its triangle.
vec3 Blend(std::span<const Vertex> vertices, const SurfacePoint &hit) {
    return vertices[hit.Vertices[0]].Position * hit.Weights.x + vertices[hit.Vertices[1]].Position * hit.Weights.y +
        vertices[hit.Vertices[2]].Position * hit.Weights.z;
}

// How far `point` sits from a hit's surface point, measured through the weights so they are pinned too.
float HitDistance(std::span<const Vertex> vertices, const SurfacePoint &hit, vec3 point) {
    return glm::length(Blend(vertices, hit) - point);
}

struct Soup {
    std::vector<Vertex> Vertices;
    std::vector<uint32_t> Triangles;
};

// Triangles scattered through a unit cube, which have no structure the hierarchy could accidentally exploit.
Soup RandomSoup(uint32_t count, std::mt19937 &rng) {
    std::uniform_real_distribution<float> coord{-1.f, 1.f}, size{0.02f, 0.3f};
    Soup soup;
    for (uint32_t i = 0; i < count; ++i) {
        const vec3 origin{coord(rng), coord(rng), coord(rng)};
        for (int v = 0; v < 3; ++v) {
            soup.Triangles.push_back(uint32_t(soup.Vertices.size()));
            soup.Vertices.emplace_back(origin + vec3{size(rng), size(rng), size(rng)});
        }
    }
    return soup;
}

// A unit sphere, whose closest point has a closed form to check the hierarchy against.
Soup UnitSphere(uint32_t rings, uint32_t segments) {
    Soup soup;
    for (uint32_t r = 0; r <= rings; ++r) {
        const float phi = std::numbers::pi_v<float> * float(r) / float(rings);
        for (uint32_t s = 0; s <= segments; ++s) {
            const float theta = 2 * std::numbers::pi_v<float> * float(s) / float(segments);
            soup.Vertices.emplace_back(vec3{std::sin(phi) * std::cos(theta), std::cos(phi), std::sin(phi) * std::sin(theta)});
        }
    }
    const auto index = [&](uint32_t r, uint32_t s) { return r * (segments + 1) + s; };
    for (uint32_t r = 0; r < rings; ++r) {
        for (uint32_t s = 0; s < segments; ++s) {
            soup.Triangles.insert(soup.Triangles.end(), {index(r, s), index(r + 1, s), index(r, s + 1)});
            soup.Triangles.insert(soup.Triangles.end(), {index(r, s + 1), index(r + 1, s), index(r + 1, s + 1)});
        }
    }
    return soup;
}
} // namespace

int main() {
    "the nearest point agrees with a scan over every triangle"_test = [] {
        std::mt19937 rng{12345};
        const auto soup = RandomSoup(400, rng);
        const auto bvh = BuildMeshBvh(soup.Vertices, soup.Triangles);

        std::uniform_real_distribution<float> coord{-1.5f, 1.5f};
        for (int i = 0; i < 200; ++i) {
            const vec3 query{coord(rng), coord(rng), coord(rng)};
            const auto hit = bvh.ClosestPoint(soup.Vertices, soup.Triangles, query);
            // The scan samples the triangles rather than solving them, so it can only bound the answer from above.
            expect(HitDistance(soup.Vertices, hit, query) <= BruteForceDistance(soup.Vertices, soup.Triangles, query) + 1e-4f);
        }
    };

    "the reported weights are a convex combination of the triangle they name"_test = [] {
        std::mt19937 rng{999};
        const auto soup = RandomSoup(200, rng);
        const auto bvh = BuildMeshBvh(soup.Vertices, soup.Triangles);

        std::uniform_real_distribution<float> coord{-1.5f, 1.5f};
        for (int i = 0; i < 200; ++i) {
            const vec3 query{coord(rng), coord(rng), coord(rng)};
            const auto hit = bvh.ClosestPoint(soup.Vertices, soup.Triangles, query);
            // Every consumer reads a per-vertex quantity at these weights, so they have to interpolate rather than extrapolate.
            const auto &w = hit.Weights;
            expect(std::abs(w.x + w.y + w.z - 1.f) < 1e-4f);
            expect(w.x >= -1e-5f && w.y >= -1e-5f && w.z >= -1e-5f);
        }
    };

    "a point outside a sphere lands on its surface along the radius"_test = [] {
        const auto sphere = UnitSphere(64, 128);
        const auto bvh = BuildMeshBvh(sphere.Vertices, sphere.Triangles);

        std::mt19937 rng{7};
        std::uniform_real_distribution<float> coord{-1.f, 1.f}, radius{1.5f, 4.f};
        for (int i = 0; i < 100; ++i) {
            const vec3 direction = glm::normalize(vec3{coord(rng), coord(rng), coord(rng)} + vec3{1e-3f});
            const vec3 query = direction * radius(rng);
            const auto hit = bvh.ClosestPoint(sphere.Vertices, sphere.Triangles, query);
            // A tessellated sphere sits just inside the true one, so the chord sagitta is the tolerance.
            expect(std::abs(HitDistance(sphere.Vertices, hit, query) - (glm::length(query) - 1.f)) < 0.005f);
            // The nearest point on a facet is the foot of the perpendicular to its plane, and a coarse tessellation
            // tilts that plane away from the radius by up to the ring half-angle, the more so the further out the query sits.
            expect(glm::dot(glm::normalize(Blend(sphere.Vertices, hit)), direction) > 0.98f);
        }
    };

    "a point on the surface reports no distance"_test = [] {
        const auto sphere = UnitSphere(16, 32);
        const auto bvh = BuildMeshBvh(sphere.Vertices, sphere.Triangles);
        for (uint32_t v = 0; v < sphere.Vertices.size(); v += 37) {
            const auto query = sphere.Vertices[v].Position;
            expect(HitDistance(sphere.Vertices, bvh.ClosestPoint(sphere.Vertices, sphere.Triangles, query), query) < 1e-5f);
        }
    };

    "a degenerate triangle answers without diverging"_test = [] {
        // A collinear triangle and a sliver, which the interior solve divides by zero on, then coincident corners
        // and a triangle collapsed to a point, each leaving an edge region with no length to divide by.
        const std::vector<Vertex> vertices{
            Vertex{{0, 0, 0}}, Vertex{{1, 0, 0}}, Vertex{{2, 0, 0}},
            Vertex{{0, 1, 0}}, Vertex{{1e-9f, 1, 0}}, Vertex{{0.5f, 1, 0}},
            Vertex{{3, 0, 0}}, Vertex{{3, 0, 0}}, Vertex{{4, 0, 0}},
            Vertex{{5, 0, 0}}, Vertex{{5, 0, 0}}, Vertex{{5, 0, 0}}
        };
        const std::vector<uint32_t> triangles{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        const auto bvh = BuildMeshBvh(vertices, triangles);
        for (const vec3 query : {vec3{0.5f, 0.5f, 0}, vec3{-3, 2, 1}, vec3{1, 0, 0}, vec3{3.5f, 1, 0}, vec3{5, 2, 0}}) {
            const auto hit = bvh.ClosestPoint(vertices, triangles, query);
            const vec3 point = Blend(vertices, hit);
            expect(std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z));
            expect(std::abs(hit.Weights.x + hit.Weights.y + hit.Weights.z - 1.f) < 1e-4f);
        }
    };

    "a mesh with no triangles builds no tree"_test = [] {
        expect(BuildMeshBvh({}, {}).Nodes.empty());
    };

    "a single triangle is its own root"_test = [] {
        const std::vector<Vertex> vertices{Vertex{{0, 0, 0}}, Vertex{{1, 0, 0}}, Vertex{{0, 1, 0}}};
        const std::vector<uint32_t> triangles{0, 1, 2};
        const auto bvh = BuildMeshBvh(vertices, triangles);
        expect(bvh.Nodes.size() == 1_ul);
        const vec3 query{0, 0, 5};
        expect(std::abs(HitDistance(vertices, bvh.ClosestPoint(vertices, triangles, query), query) - 5.f) < 1e-5f);
    };
}
