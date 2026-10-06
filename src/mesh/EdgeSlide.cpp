#include "mesh/EdgeSlide.h"
#include "Profile.h"
#include "mesh/MeshEdgeUsers.h"
#include "mesh/MeshStore.h"
#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace {
constexpr float Epsilon = std::numeric_limits<float>::epsilon();
vec3 Unit(vec3 v) {
    const float l = Length(v);
    return l > 0.f ? v / l : vec3{};
}

vec3 FaceDestination(const Mesh &mesh, he::HH h) {
    const auto &c = mesh.GetConnectivity();
    const auto position = [&](he::HH corner) { return mesh.GetPosition(mesh.GetToVertex(corner)); };
    if (mesh.GetValence(mesh.GetFace(h)) == 4u) return position(c.Next(c.Next(h)));
    const auto origin = position(h);
    const auto normal = Unit(Unit(origin - position(c.Previous(h))) + Unit(position(c.Next(h)) - origin));
    vec3 result = (position(c.Previous(h)) + position(c.Next(h))) * .5f;
    float nearest = std::numeric_limits<float>::max();
    for (auto at = c.Next(h); at != c.Previous(h); at = c.Next(at)) {
        const auto a = position(at), delta = position(c.Next(at)) - a;
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

// Blender's forked-rail intersection, including the conical-region rejection.
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
} // namespace

std::vector<EdgeSlideDirections> PlanEdgeSlide(const MeshStore &meshes, const Mesh &mesh, vec3 direction, vec3 scale, uint32_t reference) {
    const profile::CpuScope scope{"PlanEdgeSlide"};
    if (!mesh.FaceCount()) return {};
    struct Link {
        uint32_t Vertex{InvalidOffset};
        he::EH Edge;
    };
    struct Node {
        he::VH Vertex;
        std::array<Link, 2> Links;
        uint32_t Degree{};
        bool Inner{}, Done{};
    };
    std::vector<Node> nodes;
    std::unordered_map<uint32_t, uint32_t> index;
    const auto id = mesh.GetStoreId();
    meshes.GetSelectedElements(id, Element::Vertex).ForEach([&](uint32_t v) {
        index.emplace(v, uint32_t(nodes.size()));
        nodes.push_back({.Vertex = he::VH{v}});
    });
    if (nodes.empty()) return {};
    const auto &c = mesh.GetConnectivity();
    MeshEdgeUsers users{mesh};
    bool invalid = false;
    meshes.GetSelectedElements(id, Element::Edge).ForEach([&](uint32_t e) {
        if (invalid) return;
        const auto h = mesh.GetHalfedge(he::EH{e}, 0u);
        const auto count = users.Get(h).Count;
        const auto a = index.find(*mesh.GetFromVertex(h)), b = index.find(*mesh.GetToVertex(h));
        if (count == 0u || count > 2u || a == index.end() || b == index.end() || a == b) {
            invalid = true;
            return;
        }
        for (const auto [from, to] : {std::pair{a->second, b->second}, std::pair{b->second, a->second}}) {
            auto &node = nodes[from];
            if (node.Degree == 2u) {
                invalid = true;
                return;
            }
            node.Links[node.Degree++] = {to, he::EH{e}};
        }
    });
    if (invalid) return {};
    const auto incidence = mesh.GetVertexEdgeIncidence();
    for (auto &node : nodes) {
        if (!node.Degree) return {};
        uint32_t edge_count = 0u;
        bool boundary = false;
        for (const auto e : incidence.Incident(*node.Vertex)) {
            ++edge_count;
            boundary |= users.Get(mesh.GetHalfedge(he::EH{e}, 0u)).Count == 1u;
        }
        node.Inner = edge_count == 2u && !boundary;
    }
    struct Rail {
        he::FH Face;
        he::VH Vertex;
        vec3 Position{};
    };
    struct State {
        uint32_t Index{InvalidOffset};
        he::EH Edge;
        std::array<Rail, 2> Rails;
    };
    std::vector<EdgeSlideDirections> output(nodes.size());
    const auto position = [&](uint32_t i) { return mesh.GetPosition(nodes[i].Vertex); };
    const auto next_link = [&](uint32_t i, uint32_t previous) {
        for (const auto link : nodes[i].Links)
            if (link.Vertex != InvalidOffset && link.Vertex != previous) return link;
        return Link{};
    };
    const auto boundary_rail = [&](he::VH v, he::VH destination) {
        return destination && users.Get(MeshEdgeUsers::Key(*v, *destination)).Count == 1u;
    };
    for (uint32_t seed = 0u; seed < nodes.size(); ++seed) {
        if (nodes[seed].Done) continue;
        // Walk to a chain endpoint, or return to the seed for a closed loop.
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
            if (next_index != InvalidOffset) next.Edge = next_link(next_index, curr.Index).Edge;
            if (next_index != InvalidOffset) {
                const auto old = curr;
                const auto edge_users = users.Get(mesh.GetHalfedge(curr.Edge, 0u));
                for (const auto h_value : {edge_users.First, edge_users.Second}) {
                    if (h_value == InvalidOffset) continue;
                    const he::HH edge_corner{h_value};
                    const auto face = mesh.GetFace(edge_corner);
                    const bool to_current = mesh.GetToVertex(edge_corner) == nodes[curr.Index].Vertex;
                    const auto current_corner = to_current ? edge_corner : c.Previous(edge_corner);
                    const auto next_corner = to_current ? c.Previous(edge_corner) : edge_corner;
                    const auto target_corner = to_current ? c.Next(current_corner) : c.Previous(current_corner);
                    const auto next_target = to_current ? c.Previous(next_corner) : c.Next(next_corner);
                    const auto target = mesh.GetToVertex(target_corner);
                    const auto next_edge = to_current ? mesh.GetEdge(next_corner) : mesh.GetEdge(c.Next(next_corner));
                    bool intersect = false;
                    int side = -1;
                    for (int s = 0; s < 2; ++s)
                        if (face == old.Rails[s].Face || target == old.Rails[s].Vertex) {
                            side = s;
                            break;
                        }
                    if (side == -1 && (old.Rails[0].Face || old.Rails[1].Face)) {
                        // Walk the face sector around this vertex to the previous selected edge.
                        auto edge = to_current ? c.Next(current_corner) : current_corner;
                        for (uint32_t n = 0u; n < c.VertexCorners[*nodes[curr.Index].Vertex].y; ++n) {
                            const auto opposite = mesh.GetOppositeHalfedge(edge);
                            if (!opposite || mesh.GetFace(opposite) == face) break;
                            const auto other_face = mesh.GetFace(opposite);
                            for (int s = 0; s < 2; ++s)
                                if (other_face == old.Rails[s].Face) {
                                    side = s;
                                    intersect = true;
                                    break;
                                }
                            if (side != -1) break;
                            edge = mesh.GetToVertex(opposite) == nodes[curr.Index].Vertex ? c.Next(opposite) : c.Previous(opposite);
                        }
                    }
                    if (side == -1) {
                        if (!curr.Rails[0].Face || !curr.Rails[1].Face) side = bool(curr.Rails[0].Face);
                        else {
                            intersect = true;
                            const bool a = boundary_rail(nodes[curr.Index].Vertex, curr.Rails[0].Vertex), b = boundary_rail(nodes[curr.Index].Vertex, curr.Rails[1].Vertex);
                            const auto delta = mesh.GetPosition(target) - position(curr.Index);
                            side = a != b ? int(b) : int(Dot(delta, Unit(curr.Rails[0].Position - position(curr.Index))) < Dot(delta, Unit(curr.Rails[1].Position - position(curr.Index))));
                        }
                    }
                    const auto destination = mesh.GetPosition(target);
                    if (!curr.Rails[side].Face) curr.Rails[side] = {face, nodes[curr.Index].Inner ? he::VH{} : target, nodes[curr.Index].Inner ? FaceDestination(mesh, current_corner) : destination};
                    const bool across_face = next.Edge == next_edge || nodes[next_index].Inner;
                    next.Rails[side] = {face, across_face ? he::VH{} : mesh.GetToVertex(next_target), across_face ? FaceDestination(mesh, next_corner) : mesh.GetPosition(mesh.GetToVertex(next_target))};
                    if (intersect && prev.Index != InvalidOffset) curr.Rails[side].Position = RailIntersection(position(curr.Index), prev.Rails[side].Position, curr.Rails[side].Position, destination, next.Rails[side].Position);
                }
            }
            auto &dirs = output[curr.Index];
            dirs.Positive = curr.Rails[0].Face ? curr.Rails[0].Position - position(curr.Index) : vec3{};
            dirs.Negative = curr.Rails[1].Face ? curr.Rails[1].Position - position(curr.Index) : vec3{};
            prev = curr;
            curr = next;
        }
        uint32_t reference_index = chain.front();
        for (const auto i : chain)
            if (*nodes[i].Vertex == reference) reference_index = i;
        const auto &ref = output[reference_index];
        if (Dot(Unit(ref.Positive * scale), direction) < Dot(Unit(ref.Negative * scale), direction))
            for (const auto i : chain) std::swap(output[i].Positive, output[i].Negative);
    }
    return output;
}
