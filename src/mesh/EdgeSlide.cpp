#include "mesh/EdgeSlide.h"
#include "Profile.h"
#include "mesh/EdgeKey.h"
#include "mesh/Mesh.h"
#include "mesh/Neighborhood.h"
#include "numeric/VectorMath.h"

namespace {
constexpr float Epsilon = std::numeric_limits<float>::epsilon();
vec3 Unit(vec3 v) {
    const float l = Length(v);
    return l > 0.f ? v / l : vec3{};
}

vec3 FaceDestination(const LocalMesh &local, uint32_t h) {
    const auto position = [&](uint32_t corner) { return local.Position(local.At(corner).Vertex); };
    const auto &corner = local.At(h);
    if (local.Valence(corner.Face) == 4u) return position(local.At(corner.Next).Next);
    const auto origin = position(h);
    const auto normal = Unit(Unit(origin - position(corner.Previous)) + Unit(position(corner.Next) - origin));
    vec3 result = (position(corner.Previous) + position(corner.Next)) * .5f;
    float nearest = std::numeric_limits<float>::max();
    for (auto at = corner.Next; at != corner.Previous; at = local.At(at).Next) {
        const auto a = position(at), delta = position(local.At(at).Next) - a;
        const float denom = Dot(delta, normal);
        if (denom == 0.f) continue;
        const float t = Dot(origin - a, normal) / denom;
        if (t <= -Epsilon || t >= 1.f + Epsilon) continue;
        const auto p = a + delta * t;
        const float distance = Dot(p - origin, p - origin);
        if (distance < nearest) {
            nearest = distance;
            result = p;
        }
    }
    return result;
}

vec3 RailIntersection(vec3 before, vec3 a, vec3 b, vec3 c, vec3 d) {
    const auto u = b - a, v = d - c, w = a - c;
    const float aa = Dot(u, u), bb = Dot(u, v), cc = Dot(v, v), dd = Dot(u, w), ee = Dot(v, w);
    const float determinant = aa * cc - bb * bb;
    const auto fallback = (b + c) * .5f;
    if (aa <= Epsilon * Epsilon || cc <= Epsilon * Epsilon || std::abs(determinant) <= Epsilon * Epsilon) return fallback;
    const auto p = a + u * ((bb * ee - cc * dd) / determinant), q = c + v * ((aa * ee - bb * dd) / determinant);
    const auto dir1 = Unit(b - before), dir2 = Unit(c - before), cross = Cross(dir1, dir2);
    if (Length(cross) < Epsilon) return fallback;
    const auto n = Unit(cross), plane1 = Unit(Cross(n, dir1)), plane2 = Unit(Cross(dir2, n));
    if (Length(plane1) < Epsilon || Length(plane2) < Epsilon) return fallback;
    if (Dot(p - before, plane1) <= 0.f || Dot(p - before, plane2) <= 0.f || Dot(q - before, plane1) <= 0.f || Dot(q - before, plane2) <= 0.f) return fallback;
    return (p + q) * .5f;
}

std::vector<uint32_t> SidesOf(const LocalMesh &local, uint32_t a, uint32_t b) {
    std::vector<uint32_t> sides;
    const auto key = EdgeKey(a, b);
    for (const auto &item : local.Fan(std::min(a, b))) {
        if (item.Face == InvalidOffset) continue;
        for (const auto side : {item.Corner, local.At(item.Corner).Next}) {
            const auto &corner = local.At(side);
            if (EdgeKey(corner.From, corner.Vertex) == key && corner.From != corner.Vertex) sides.push_back(side);
        }
    }
    return sides;
}
} // namespace

std::vector<EdgeSlideDirections> PlanEdgeSlide(const MeshStore &meshes, const MeshPipelines &pipelines, const Mesh &mesh, const GeometrySelection &selection, vec3 direction, vec3 scale, uint32_t reference) {
    const profile::CpuScope scope{"PlanEdgeSlide"};
    if (!mesh.FaceCount()) return {};
    struct Link {
        uint32_t Vertex{InvalidOffset};
        uint32_t Edge{InvalidOffset};
    };
    struct Node {
        uint32_t Vertex;
        std::array<Link, 2> Links;
        uint32_t Degree{};
        bool Inner{}, Done{};
    };
    std::vector<Node> nodes;
    std::unordered_map<uint32_t, uint32_t> index;
    for (const auto v : selection.Vertices) {
        index.emplace(v, uint32_t(nodes.size()));
        nodes.push_back({.Vertex = v});
    }
    if (nodes.empty()) return {};
    const LocalMesh local{meshes, pipelines, mesh.GetStoreId(), selection.Vertices};
    std::unordered_map<uint32_t, std::array<uint32_t, 2>> ends;
    for (const auto v : selection.Vertices) {
        for (const auto &edge : local.Edges(v)) ends.try_emplace(edge.Edge, std::array{v, edge.Other});
    }
    for (const auto e : selection.Edges) {
        const auto found = ends.find(e);
        if (found == ends.end()) return {};
        const auto [from_vertex, to_vertex] = found->second;
        const auto count = local.Sides(from_vertex, to_vertex);
        const auto a = index.find(from_vertex), b = index.find(to_vertex);
        if (count == 0u || count > 2u || a == index.end() || b == index.end() || a == b) return {};
        for (const auto [from, to] : {std::pair{a->second, b->second}, std::pair{b->second, a->second}}) {
            auto &node = nodes[from];
            if (node.Degree == 2u) return {};
            node.Links[node.Degree++] = {to, e};
        }
    }
    for (auto &node : nodes) {
        if (!node.Degree) return {};
        const auto edges = local.Edges(node.Vertex);
        node.Inner = edges.size() == 2u && std::ranges::none_of(edges, [](const auto &e) { return e.Sides == 1u; });
    }
    struct Rail {
        uint32_t Face{InvalidOffset};
        uint32_t Vertex{InvalidOffset};
        vec3 Position{};
    };
    struct State {
        uint32_t Index{InvalidOffset};
        uint32_t Edge{InvalidOffset};
        std::array<Rail, 2> Rails;
    };
    std::vector<EdgeSlideDirections> output(nodes.size());
    const auto position = [&](uint32_t i) { return local.Position(nodes[i].Vertex); };
    const auto next_link = [&](uint32_t i, uint32_t previous) {
        for (const auto link : nodes[i].Links) {
            if (link.Vertex != InvalidOffset && link.Vertex != previous) return link;
        }
        return Link{};
    };
    const auto boundary_rail = [&](uint32_t v, uint32_t destination) {
        return destination != InvalidOffset && local.Sides(v, destination) == 1u;
    };
    for (uint32_t seed = 0u; seed < nodes.size(); ++seed) {
        if (nodes[seed].Done) continue;
        uint32_t start = seed, previous = InvalidOffset;
        while (nodes[start].Degree == 2u) {
            const auto next = next_link(start, previous).Vertex;
            previous = start;
            start = next;
            if (start == seed) break;
        }
        const bool closed = nodes[start].Degree == 2u;
        previous = InvalidOffset;
        std::vector<uint32_t> chain;
        uint32_t at = start;
        do {
            chain.push_back(at);
            nodes[at].Done = true;
            const auto next = next_link(at, previous).Vertex;
            previous = at;
            at = next;
        } while (at != InvalidOffset && at != start);
        State prev, curr{.Index = start, .Edge = next_link(start, InvalidOffset).Edge};
        for (uint32_t step = 0u; step < chain.size() + uint32_t(closed); ++step) {
            const uint32_t next_index = step + 1u < chain.size() ? chain[step + 1u] : closed ? chain[(step + 1u) % chain.size()] :
                                                                                               InvalidOffset;
            State next{.Index = next_index};
            if (next_index != InvalidOffset) {
                next.Edge = next_link(next_index, curr.Index).Edge;
                const auto old = curr;
                const auto vertex = nodes[curr.Index].Vertex;
                for (const auto edge_corner : SidesOf(local, vertex, nodes[next_index].Vertex)) {
                    const auto face = local.At(edge_corner).Face;
                    const bool to_current = local.At(edge_corner).Vertex == vertex;
                    const auto current_corner = to_current ? edge_corner : local.At(edge_corner).Previous;
                    const auto next_corner = to_current ? local.At(edge_corner).Previous : edge_corner;
                    const auto target_corner = to_current ? local.At(current_corner).Next : local.At(current_corner).Previous;
                    const auto next_target = to_current ? local.At(next_corner).Previous : local.At(next_corner).Next;
                    const auto target = local.At(target_corner).Vertex;
                    const auto next_edge = to_current ? local.At(next_corner).Edge : local.At(local.At(next_corner).Next).Edge;
                    bool intersect = false;
                    int side = -1;
                    for (int s = 0; s < 2; ++s) {
                        if (face == old.Rails[s].Face || target == old.Rails[s].Vertex) {
                            side = s;
                            break;
                        }
                    }
                    if (side == -1 && (old.Rails[0].Face != InvalidOffset || old.Rails[1].Face != InvalidOffset)) {
                        auto edge = to_current ? local.At(current_corner).Next : current_corner;
                        for (uint32_t n = 0u; n < local.Fan(vertex).size(); ++n) {
                            const auto opposite = local.At(edge).Opposite;
                            if (opposite == InvalidOffset || local.At(opposite).Face == face) break;
                            const auto other_face = local.At(opposite).Face;
                            for (int s = 0; s < 2; ++s) {
                                if (other_face == old.Rails[s].Face) {
                                    side = s;
                                    intersect = true;
                                    break;
                                }
                            }
                            if (side != -1) break;
                            edge = local.At(opposite).Vertex == vertex ? local.At(opposite).Next : local.At(opposite).Previous;
                        }
                    }
                    if (side == -1) {
                        if (curr.Rails[0].Face == InvalidOffset || curr.Rails[1].Face == InvalidOffset) side = curr.Rails[0].Face != InvalidOffset;
                        else {
                            intersect = true;
                            const bool a = boundary_rail(vertex, curr.Rails[0].Vertex), b = boundary_rail(vertex, curr.Rails[1].Vertex);
                            const auto delta = local.Position(target) - position(curr.Index);
                            side = a != b ? int(b) : int(Dot(delta, Unit(curr.Rails[0].Position - position(curr.Index))) < Dot(delta, Unit(curr.Rails[1].Position - position(curr.Index))));
                        }
                    }
                    const auto destination = local.Position(target);
                    if (curr.Rails[side].Face == InvalidOffset) curr.Rails[side] = {face, nodes[curr.Index].Inner ? InvalidOffset : target, nodes[curr.Index].Inner ? FaceDestination(local, current_corner) : destination};
                    const bool across_face = next.Edge == next_edge || nodes[next_index].Inner;
                    next.Rails[side] = {face, across_face ? InvalidOffset : local.At(next_target).Vertex, across_face ? FaceDestination(local, next_corner) : local.Position(local.At(next_target).Vertex)};
                    if (intersect && prev.Index != InvalidOffset) curr.Rails[side].Position = RailIntersection(position(curr.Index), prev.Rails[side].Position, curr.Rails[side].Position, destination, next.Rails[side].Position);
                }
            }
            auto &dirs = output[curr.Index];
            dirs.Positive = curr.Rails[0].Face != InvalidOffset ? curr.Rails[0].Position - position(curr.Index) : vec3{};
            dirs.Negative = curr.Rails[1].Face != InvalidOffset ? curr.Rails[1].Position - position(curr.Index) : vec3{};
            prev = curr;
            curr = next;
        }
        uint32_t reference_index = chain.front();
        for (const auto i : chain) {
            if (nodes[i].Vertex == reference) reference_index = i;
        }
        const auto &ref = output[reference_index];
        if (Dot(Unit(ref.Positive * scale), direction) < Dot(Unit(ref.Negative * scale), direction)) {
            for (const auto i : chain) std::swap(output[i].Positive, output[i].Negative);
        }
    }
    return output;
}
