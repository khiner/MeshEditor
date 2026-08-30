#include "mesh/Tets.h"

#include "numeric/Predicates.h"

#include "meshoptimizer.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <unordered_map>

namespace {
// Geometric steps the staged rebuild takes to reach the target.
constexpr int Stages{4};
// Freeze-and-retry attempts per stage, after which the stage's input is kept.
constexpr int MaxRounds{6};
// Frozen ball radius around a defect, in multiples of the length the defect measures.
constexpr double LockRadius{0.5};
// The collinearity tolerance the tetrahedralizer starts an edge recovery from.
constexpr double CollinearAngTol{179.9};
// The tetrahedralizer loosens that tolerance once, by a 180th of the angle's excess over it.
// A vertex inside an edge within this many degrees of straight carries the loosened tolerance past 180 degrees, and the surface comes back rejected.
constexpr double StraightTol{(180 - CollinearAngTol) / 181};
const double SinStraightTol{std::sin(StraightTol * std::numbers::pi / 180)};

// True when segment (p, q) passes through triangle (a, b, c).
// Strict rejects contact at a shared vertex, for triangles that meet at one corner.
bool SegmentCrossesTriangle(const dvec3 &p, const dvec3 &q, const dvec3 &a, const dvec3 &b, const dvec3 &c, bool strict) {
    const double sp = geom::Orient3D(a, b, c, p), sq = geom::Orient3D(a, b, c, q);
    if (strict ? !((sp > 0 && sq < 0) || (sp < 0 && sq > 0)) : ((sp > 0 && sq > 0) || (sp < 0 && sq < 0) || (sp == 0 && sq == 0))) return false;
    const double s1 = geom::Orient3D(p, q, a, b), s2 = geom::Orient3D(p, q, b, c), s3 = geom::Orient3D(p, q, c, a);
    if (strict) return (s1 > 0 && s2 > 0 && s3 > 0) || (s1 < 0 && s2 < 0 && s3 < 0);
    return (s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0);
}

// Triangles sharing an edge always meet cleanly.
// Triangles sharing one vertex intersect when the edges opposite that vertex pass through them.
bool TrianglesIntersect(const uint32_t *ta, const uint32_t *tb, const dvec3 *p) {
    int shared = 0, corner_a = 0, corner_b = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (ta[i] == tb[j]) {
                ++shared;
                corner_a = i;
                corner_b = j;
            }
        }
    }
    if (shared >= 2) return false;
    const dvec3 a[3]{p[ta[0]], p[ta[1]], p[ta[2]]}, b[3]{p[tb[0]], p[tb[1]], p[tb[2]]};
    if (shared == 1) {
        return SegmentCrossesTriangle(a[(corner_a + 1) % 3], a[(corner_a + 2) % 3], b[0], b[1], b[2], true) ||
            SegmentCrossesTriangle(b[(corner_b + 1) % 3], b[(corner_b + 2) % 3], a[0], a[1], a[2], true);
    }
    for (int e = 0; e < 3; ++e) {
        if (SegmentCrossesTriangle(a[e], a[(e + 1) % 3], b[0], b[1], b[2], false)) return true;
        if (SegmentCrossesTriangle(b[e], b[(e + 1) % 3], a[0], a[1], a[2], false)) return true;
    }
    return false;
}

// Uniform spatial hash over cells of a fixed size.
struct Grid {
    double Cell;
    dvec3 Min;
    std::unordered_map<int64_t, std::vector<uint32_t>> Cells;

    static int64_t Key(int64_t x, int64_t y, int64_t z) { return (x * 73856093) ^ (y * 19349663) ^ (z * 83492791); }

    void ForCellsIn(const dvec3 &lo, const dvec3 &hi, auto &&visit) {
        const dvec3 c0 = numeric::Floor((lo - Min) / Cell), c1 = numeric::Floor((hi - Min) / Cell);
        for (auto x = int64_t(c0.x); x <= int64_t(c1.x); ++x) {
            for (auto y = int64_t(c0.y); y <= int64_t(c1.y); ++y) {
                for (auto z = int64_t(c0.z); z <= int64_t(c1.z); ++z) visit(Cells[Key(x, y, z)]);
            }
        }
    }
};

// A neighbourhood of the simplified surface to freeze, as a center and the length its lock radius scales with.
struct Defect {
    dvec3 Center;
    double Scale;
};

// A triangle, centred on its centroid and measured by its longest edge.
Defect TriangleDefect(const dvec3 &v0, const dvec3 &v1, const dvec3 &v2) {
    return {(v0 + v1 + v2) / 3.0, std::max({numeric::Length(v1 - v0), numeric::Length(v2 - v1), numeric::Length(v0 - v2)})};
}

// Triangles that pass through each other.
void FindFolds(const std::vector<dvec3> &points, const std::vector<uint32_t> &tris, std::vector<Defect> &out) {
    const uint32_t n = uint32_t(tris.size() / 3);
    if (n == 0) return;
    // Triangle bounds stay in float, which halves what the candidate scan reads and is exact because the points came from float positions.
    std::vector<vec3> tri_min(n), tri_max(n);
    dvec3 min = points[tris[0]], max = min;
    double diagonal_sum = 0;
    for (uint32_t t = 0; t < n; ++t) {
        dvec3 lo = points[tris[t * 3]], hi = lo;
        for (uint32_t k = 1; k < 3; ++k) {
            lo = numeric::Min(lo, points[tris[t * 3 + k]]);
            hi = numeric::Max(hi, points[tris[t * 3 + k]]);
        }
        tri_min[t] = vec3{lo};
        tri_max[t] = vec3{hi};
        min = numeric::Min(min, lo);
        max = numeric::Max(max, hi);
        diagonal_sum += numeric::Length(hi - lo);
    }
    // Cells the size of an average triangle keep each bucket to a handful of candidates.
    Grid grid{std::max(diagonal_sum / n, 1e-12), min, {}};
    // A triangle spanning several cells meets the same candidate in each of them, so each one is stamped with the triangle it was last weighed against.
    std::vector<uint32_t> stamp(n, 0);
    uint32_t epoch = 0;
    for (uint32_t t = 0; t < n; ++t) {
        ++epoch;
        const vec3 lo = tri_min[t], hi = tri_max[t];
        grid.ForCellsIn(dvec3{lo}, dvec3{hi}, [&](std::vector<uint32_t> &bucket) {
            for (const uint32_t other : bucket) {
                if (stamp[other] == epoch) continue;
                stamp[other] = epoch;
                // Nearly every candidate shares a cell without overlapping, and disjoint bounds settle it without a predicate.
                if (lo.x > tri_max[other].x || hi.x < tri_min[other].x || lo.y > tri_max[other].y || hi.y < tri_min[other].y || lo.z > tri_max[other].z || hi.z < tri_min[other].z) continue;
                if (!TrianglesIntersect(&tris[t * 3], &tris[other * 3], points.data())) continue;
                for (const uint32_t f : {t, other}) {
                    out.push_back(TriangleDefect(points[tris[f * 3]], points[tris[f * 3 + 1]], points[tris[f * 3 + 2]]));
                }
            }
            bucket.push_back(t);
        });
    }
}

// Vertices that sit inside an edge they are not part of, so nearly straight that the tetrahedralizer
// gives up on recovering that edge and reports the surface as intersecting itself.
void FindVerticesInsideEdges(const std::vector<dvec3> &points, const std::vector<uint32_t> &tris, std::vector<Defect> &out) {
    std::vector<uint64_t> edges;
    edges.reserve(tris.size());
    for (size_t i = 0; i < tris.size(); i += 3) {
        for (uint32_t e = 0; e < 3; ++e) {
            const auto [lo, hi] = std::minmax(tris[i + e], tris[i + (e + 1) % 3]);
            edges.push_back((uint64_t(lo) << 32) | hi);
        }
    }
    std::ranges::sort(edges);
    edges.erase(std::ranges::unique(edges).begin(), edges.end());
    if (edges.empty()) return;

    dvec3 min = points[uint32_t(edges[0] >> 32)];
    double edge_sum = 0;
    for (const uint64_t e : edges) {
        const dvec3 &a = points[uint32_t(e >> 32)], &b = points[uint32_t(e)];
        min = numeric::Min(min, numeric::Min(a, b));
        edge_sum += numeric::Length(b - a);
    }
    // Cells the size of an average edge keep each bucket to a handful of candidates.
    Grid grid{std::max(edge_sum / double(edges.size()), 1e-12), min, {}};
    // A vertex lands in one cell, so an edge meets each candidate once.
    std::vector<unsigned char> used(points.size(), 0);
    for (const uint32_t v : tris) used[v] = 1;
    for (uint32_t v = 0; v < points.size(); ++v) {
        if (used[v]) grid.ForCellsIn(points[v], points[v], [&](std::vector<uint32_t> &bucket) { bucket.push_back(v); });
    }

    for (const uint64_t e : edges) {
        const uint32_t a = uint32_t(e >> 32), b = uint32_t(e);
        const dvec3 &pa = points[a], &pb = points[b];
        grid.ForCellsIn(numeric::Min(pa, pb), numeric::Max(pa, pb), [&](std::vector<uint32_t> &bucket) {
            for (const uint32_t v : bucket) {
                if (v == a || v == b) continue;
                // A negative dot puts the vertex between the two ends, where the sine of the angle
                // it makes there falls as that angle approaches straight.
                const dvec3 u = pa - points[v], w = pb - points[v];
                if (numeric::Dot(u, w) >= 0) continue;
                if (numeric::Length(numeric::Cross(u, w)) <= numeric::Length(u) * numeric::Length(w) * SinStraightTol) {
                    // Half the edge length around its midpoint reaches every vertex the edge was collapsed over.
                    out.emplace_back(0.5 * (pa + pb), numeric::Length(pb - pa));
                }
            }
        });
    }
}

// Everything about a simplified surface that stops it tetrahedralizing, as neighbourhoods to freeze.
std::vector<Defect> FindDefects(const std::vector<dvec3> &points, const std::vector<uint32_t> &tris) {
    std::vector<Defect> defects;
    FindFolds(points, tris, defects);
    FindVerticesInsideEdges(points, tris, defects);
    return defects;
}

// Rebuild the simplification in stages, retrying any stage that leaves a defect with the neighbourhood of each defect frozen.
// A stage only has to undo its own defects, so a frozen region costs the resolution of one stage rather than of the whole simplification.
std::vector<uint32_t> SimplifyWithoutDefects(const std::vector<dvec3> &points, const std::vector<vec3> &positions, const std::vector<uint32_t> &triangle_indices, float ratio) {
    dvec3 min = points[0], max = points[0];
    for (const auto &p : points) {
        min = numeric::Min(min, p);
        max = numeric::Max(max, p);
    }
    double edge_sum = 0;
    for (size_t i = 0; i < triangle_indices.size(); i += 3) edge_sum += numeric::Length(points[triangle_indices[i]] - points[triangle_indices[i + 1]]);
    Grid vertex_grid{std::max(edge_sum * 3 / double(triangle_indices.size()), 1e-12), min, {}};
    for (uint32_t v = 0; v < points.size(); ++v) {
        vertex_grid.ForCellsIn(points[v], points[v], [&](std::vector<uint32_t> &bucket) { bucket.push_back(v); });
    }

    const size_t base_indices = triangle_indices.size();
    std::vector<uint32_t> tris = triangle_indices;
    for (int stage = 1; stage <= Stages; ++stage) {
        const double stage_ratio = std::pow(double(ratio), double(stage) / Stages);
        const auto target_indices = std::max<size_t>(size_t(base_indices * stage_ratio) / 3 * 3, 12);
        if (target_indices >= tris.size()) continue;
        // Vertices frozen here keep this stage's collapses out of a region that came back defective.
        std::vector<unsigned char> locks(points.size(), 0);
        size_t locked = 0;
        double radius = LockRadius;
        for (int round = 0; round <= MaxRounds; ++round) {
            std::vector<uint32_t> simplified(tris.size());
            simplified.resize(meshopt_simplifyWithAttributes(simplified.data(), tris.data(), tris.size(), &positions[0].x, positions.size(), sizeof(vec3), nullptr, 0, nullptr, 0, locks.data(), target_indices, 0.05f, meshopt_SimplifySparse, nullptr));
            const auto defects = FindDefects(points, simplified);
            if (defects.empty()) {
                tris = std::move(simplified);
                break;
            }
            // Out of attempts: keep the finer mesh this stage started from and let the next stage try.
            if (round == MaxRounds) break;
            for (const auto &[center, scale] : defects) {
                const double r = radius * scale;
                vertex_grid.ForCellsIn(center - dvec3{r}, center + dvec3{r}, [&](std::vector<uint32_t> &bucket) {
                    for (const uint32_t v : bucket) {
                        if (numeric::Length(points[v] - center) <= r) locks[v] = 1;
                    }
                });
            }
            // A round that freezes nothing new repeats itself, so widen the neighbourhood.
            const auto now_locked = size_t(std::ranges::count(locks, 1));
            if (now_locked == locked) radius *= 2;
            locked = now_locked;
        }
    }
    return tris;
}
} // namespace

void SimplifySurface(std::vector<vec3> &positions, std::vector<uint32_t> &triangle_indices, float ratio) {
    if (ratio >= 1) return;

    // Simplified indices address the original vertices, so all defects are measured against these points.
    const std::vector<dvec3> points(positions.begin(), positions.end());

    // Only a thin-walled surface comes out defective from collapsing straight to the target, so try that first and pay for the staged rebuild where it does.
    const auto target_indices = std::max<size_t>(size_t(triangle_indices.size() * ratio) / 3 * 3, 12);
    std::vector<uint32_t> tris(triangle_indices.size());
    tris.resize(meshopt_simplify(tris.data(), triangle_indices.data(), triangle_indices.size(), &positions[0].x, positions.size(), sizeof(vec3), target_indices, 0.05f, 0, nullptr));
    if (!FindDefects(points, tris).empty()) tris = SimplifyWithoutDefects(points, positions, triangle_indices, ratio);

    positions.resize(meshopt_optimizeVertexFetch(positions.data(), tris.data(), tris.size(), positions.data(), positions.size(), sizeof(vec3)));
    triangle_indices = std::move(tris);
}

std::expected<tetra::Result, std::string> GenerateTets(std::vector<vec3> positions, std::vector<uint32_t> triangle_indices, tetra::Options options) {
    const std::vector<dvec3> points(positions.begin(), positions.end());
    return tetra::Tetrahedralize(points, triangle_indices, options);
}

TetMeshData BuildTetMeshData(const TetMesh &tets, vec3 scale) {
    const dvec3 inv_scale{1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z};
    TetMeshData out;
    out.Positions.reserve(tets.Points.size());
    for (const auto &p : tets.Points) out.Positions.emplace_back(p * inv_scale);
    std::vector<uint64_t> edges;
    edges.reserve(tets.Tets.size() * 6);
    for (const auto &t : tets.Tets) {
        for (uint32_t i = 0; i < 4; ++i) {
            for (uint32_t j = i + 1; j < 4; ++j) {
                const auto [lo, hi] = std::minmax(t[i], t[j]);
                edges.push_back((uint64_t(lo) << 32) | hi);
            }
        }
    }
    std::ranges::sort(edges);
    edges.erase(std::ranges::unique(edges).begin(), edges.end());
    out.EdgeIndices.reserve(edges.size() * 2);
    for (const uint64_t e : edges) {
        out.EdgeIndices.push_back(uint32_t(e >> 32));
        out.EdgeIndices.push_back(uint32_t(e));
    }
    return out;
}
