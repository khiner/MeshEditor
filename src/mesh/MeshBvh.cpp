#include "MeshBvh.h"

#include <algorithm>
#include <limits>
#include <numeric>

namespace {
AABB Union(const AABB &a, const AABB &b) { return {numeric::Min(a.Min, b.Min), numeric::Max(a.Max, b.Max)}; }
vec3 Center(const AABB &b) { return (b.Min + b.Max) * 0.5f; }

uint32_t LongestAxis(const AABB &b) {
    const vec3 extent = b.Max - b.Min;
    if (extent.x > extent.y && extent.x > extent.z) return 0;
    return extent.y > extent.z ? 1 : 2;
}

// Squared distance from a point to a box, zero for a point inside it.
float DistanceSquared(const AABB &b, vec3 p) {
    const vec3 outside = numeric::Max(numeric::Max(b.Min - p, p - b.Max), vec3{0});
    return numeric::Dot(outside, outside);
}

// Split `indices` at the median box centre along the enclosing box's longest axis, emitting children before their parent.
// Returns the node index of the subtree's root.
uint32_t Build(MeshBvh &bvh, std::span<const AABB> boxes, std::span<uint32_t> indices) {
    if (indices.size() == 1) {
        bvh.Nodes.emplace_back(boxes[indices.front()], indices.front(), MeshBvh::Node::NoChild);
        return uint32_t(bvh.Nodes.size() - 1);
    }
    AABB box;
    for (const auto i : indices) box = Union(box, boxes[i]);
    const auto axis = LongestAxis(box);
    const auto mid = indices.size() / 2;
    std::nth_element(indices.begin(), indices.begin() + mid, indices.end(), [&](uint32_t a, uint32_t b) {
        return Center(boxes[a])[axis] < Center(boxes[b])[axis];
    });
    const auto left = Build(bvh, boxes, indices.subspan(0, mid));
    const auto right = Build(bvh, boxes, indices.subspan(mid));
    bvh.Nodes.emplace_back(box, left, right);
    return uint32_t(bvh.Nodes.size() - 1);
}

} // namespace

// By Ericson's region test: the nearest point lies in the interior, on an edge, or at a vertex.
// Each region is decided by a pair of dot products.
TrianglePoint ClosestPointOnTriangle(vec3 p, vec3 a, vec3 b, vec3 c) {
    const vec3 ab = b - a, ac = c - a;
    const vec3 ap = p - a, bp = p - b, cp = p - c;
    const float d1 = numeric::Dot(ab, ap), d2 = numeric::Dot(ac, ap);
    const float d3 = numeric::Dot(ab, bp), d4 = numeric::Dot(ac, bp);
    const float d5 = numeric::Dot(ab, cp), d6 = numeric::Dot(ac, cp);

    if (d1 <= 0 && d2 <= 0) return {a, {1, 0, 0}};
    if (d3 >= 0 && d4 <= d3) return {b, {0, 1, 0}};
    if (d6 >= 0 && d5 <= d6) return {c, {0, 0, 1}};

    const float va = d3 * d6 - d5 * d4, vb = d5 * d2 - d1 * d6, vc = d1 * d4 - d3 * d2;
    // Zero-length edges defer to a nonzero edge; a point-degenerate triangle reaches the fallback below.
    const float ab_len2 = d1 - d3, ac_len2 = d2 - d6, bc_len2 = (d4 - d3) + (d5 - d6);
    if (vc <= 0 && d1 >= 0 && d3 <= 0 && ab_len2 > 0) {
        const float v = d1 / ab_len2;
        return {a + ab * v, {1 - v, v, 0}};
    }
    if (vb <= 0 && d2 >= 0 && d6 <= 0 && ac_len2 > 0) {
        const float w = d2 / ac_len2;
        return {a + ac * w, {1 - w, 0, w}};
    }
    if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0 && bc_len2 > 0) {
        const float w = (d4 - d3) / bc_len2;
        return {b + (c - b) * w, {0, 1 - w, w}};
    }

    const float sum = va + vb + vc;
    if (sum <= 0) return {a, {1, 0, 0}}; // Degenerate triangle, whose interior the region tests cannot reach.
    const float v = vb / sum, w = vc / sum;
    return {a + ab * v + ac * w, {1 - v - w, v, w}};
}

MeshBvh BuildMeshBvh(std::span<const Vertex> vertices, std::span<const uint32_t> triangle_indices) {
    MeshBvh bvh;
    const auto count = triangle_indices.size() / 3;
    if (count == 0) return bvh;

    std::vector<AABB> boxes;
    boxes.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        AABB box;
        for (size_t k = 0; k < 3; ++k) {
            const auto p = vertices[triangle_indices[i * 3 + k]].Position;
            box = {numeric::Min(box.Min, p), numeric::Max(box.Max, p)};
        }
        boxes.push_back(box);
    }

    std::vector<uint32_t> order(count);
    std::iota(order.begin(), order.end(), 0u);
    bvh.Nodes.reserve(2 * count);
    Build(bvh, boxes, order);
    return bvh;
}

SurfacePoint MeshBvh::ClosestPoint(std::span<const Vertex> vertices, std::span<const uint32_t> triangle_indices, vec3 point) const {
    SurfacePoint best;
    float best_distance2 = std::numeric_limits<float>::max();
    // A median split halves the triangle count at every level, so the depth cannot exceed the width of a triangle index.
    // The descent stores at most one node per level plus one deferred sibling.
    // Each entry carries the distance its box was ordered by, which the running best prunes against.
    std::array<std::pair<uint32_t, float>, 64> stack;
    uint32_t top = 0;
    stack[top++] = {uint32_t(Nodes.size() - 1), 0.f};
    while (top > 0) {
        const auto [index, box_distance2] = stack[--top];
        if (box_distance2 >= best_distance2) continue;
        const auto &node = Nodes[index];
        if (node.IsLeaf()) {
            const std::array tri{triangle_indices[node.Left * 3], triangle_indices[node.Left * 3 + 1], triangle_indices[node.Left * 3 + 2]};
            const auto hit = ClosestPointOnTriangle(point, vertices[tri[0]].Position, vertices[tri[1]].Position, vertices[tri[2]].Position);
            const vec3 offset = hit.Position - point;
            if (const float distance2 = numeric::Dot(offset, offset); distance2 < best_distance2) {
                best_distance2 = distance2;
                best = {tri, hit.Weights};
            }
            continue;
        }
        // Descend the nearer child first, so the running best prunes the farther one as often as it can.
        const std::pair left{node.Left, DistanceSquared(Nodes[node.Left].Box, point)};
        const std::pair right{node.Right, DistanceSquared(Nodes[node.Right].Box, point)};
        const bool left_first = left.second <= right.second;
        stack[top++] = left_first ? right : left;
        stack[top++] = left_first ? left : right;
    }
    return best;
}
