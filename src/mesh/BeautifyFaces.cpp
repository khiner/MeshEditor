#include "mesh/BeautifyFaces.h"
#include "Profile.h"
#include "mesh/MeshEdgeUsers.h"
#include "mesh/MeshStore.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <set>
#include <unordered_map>

namespace {
constexpr float NoCost = std::numeric_limits<float>::max();
uint32_t Next(uint32_t h) { return h - h % 3u + (h + 1u) % 3u; }
uint32_t Prev(uint32_t h) { return h - h % 3u + (h + 2u) % 3u; }

// Blender's BM_verts_calc_rotate_beauty. Projection is expressed directly in
// the quad plane, so no basis or copied position array is needed.
float BeautyCost(vec3 a, vec3 b, vec3 c, vec3 d, bool angle) {
    const auto cross = [](vec3 a, vec3 b, vec3 c) { return Cross(b - a, c - a); };
    const auto normalize = [](vec3 p) { const float n=Length(p); return n>0.f ? p/n : vec3{}; };
    if (angle) {
        const auto old_a = normalize(cross(b, c, d)), old_b = normalize(cross(b, d, a));
        const auto new_a = normalize(cross(a, b, c)), new_b = normalize(cross(a, c, d));
        if (new_a == vec3{} || new_b == vec3{}) return NoCost;
        const auto between = [](vec3 a, vec3 b) {
            const bool acute = Dot(a, b) >= 0.f;
            const float angle = 2.f * std::asin(std::min(1.f, Length(acute ? a - b : a + b) * .5f));
            return acute ? angle : std::numbers::pi_v<float> - angle;
        };
        return between(new_a, new_b) - between(old_a, old_b);
    }
    const auto sum = cross(b, c, d) + cross(b, d, a);
    const float scale = Length(sum);
    if (!(scale > 0.f)) return NoCost;
    const auto normal = sum / scale;
    b -= a;
    c -= a;
    d -= a;
    a = {};
    b -= normal * Dot(b, normal);
    c -= normal * Dot(c, normal);
    d -= normal * Dot(d, normal);
    const float old_a = Dot(cross(b, c, d), normal), old_b = Dot(cross(b, d, a), normal);
    const float new_a = Dot(cross(a, b, c), normal), new_b = Dot(cross(a, c, d), normal);
    const auto sign = [](float x) { return x > 1e-5f ? 1 : x < -1e-5f ? -1 :
                                                                        0; };
    if (!(sign(old_a / scale) + sign(old_b / scale))) return NoCost;
    if ((new_a >= 0.f) != (new_b >= 0.f) || std::abs(new_a) <= 1e-12f || std::abs(new_b) <= 1e-12f ||
        (old_a >= 0.f) != (old_b >= 0.f)) return NoCost;
    if (std::abs(old_a) <= 1e-12f || std::abs(old_b) <= 1e-12f) return -NoCost;
    const float ab = Length(a - b), bc = Length(b - c), cd = Length(c - d), da = Length(d - a), ac = Length(a - c), bd = Length(b - d);
    const float old_quality = std::abs(old_a) / (bc + cd + bd) + std::abs(old_b) / (da + ab + bd);
    const float new_quality = std::abs(new_a) / (ab + bc + ac) + std::abs(new_b) / (cd + da + ac);
    // Equal-quality diagonals must not rotate because projection and summation
    // rounded differently. Scale the tolerance with quality, not world units.
    const float cost = old_quality - new_quality;
    return std::abs(cost) <= 8.f * std::numeric_limits<float>::epsilon() * std::max(old_quality, new_quality) ? 0.f : cost;
}

struct Planner {
    const Mesh &M;
    MeshEdgeUsers Canonical;
    struct Node {
        uint32_t Source, Edge;
    };
    struct Edge {
        uint32_t Source{}, Count{}, External{};
        std::array<uint32_t, 2> Nodes{InvalidOffset, InvalidOffset};
        bool Eligible{};
        float Cost{NoCost};
        std::set<std::array<uint32_t, 4>> States;
    };
    std::vector<Node> Nodes;
    std::vector<Edge> Edges;
    std::vector<uint32_t> Faces;
    std::unordered_map<uint64_t, uint32_t> Current, Original;
    std::set<std::pair<float, uint32_t>> Heap;
    bool Angle;
    uint32_t Rotations{};
    Planner(const Mesh &mesh, bool angle) : M{mesh}, Canonical{mesh}, Angle{angle} {}
    uint32_t Vertex(uint32_t h) const { return *M.GetToVertex(he::HH{Nodes[h].Source}); }
    uint64_t Key(uint32_t h) const { return MeshEdgeUsers::Key(Vertex(h), Vertex(Next(h))); }
    std::array<uint32_t, 4> State(uint32_t edge, bool alternate) const {
        const auto &e = Edges[edge];
        const auto h = e.Nodes[0], g = e.Nodes[1];
        uint32_t a = Vertex(h), b = Vertex(Next(h)), c = Vertex(Prev(h)), d = Vertex(Prev(g));
        if (a > b) std::swap(a, b);
        if (c > d) std::swap(c, d);
        return alternate ? std::array{c, d, a, b} : std::array{a, b, c, d};
    }
    void Update(uint32_t edge) {
        auto &e = Edges[edge];
        if (!e.Eligible) return;
        if (e.Cost < 0.f) Heap.erase({e.Cost, edge});
        e.Cost = NoCost;
        const auto h = e.Nodes[0], g = e.Nodes[1];
        if (Vertex(Prev(h)) == Vertex(Prev(g)) || e.States.contains(State(edge, true))) return;
        const auto p = [&](uint32_t h) { return M.GetPosition(he::VH{Vertex(h)}); };
        e.Cost = BeautyCost(p(Prev(h)), p(h), p(Prev(g)), p(Next(h)), Angle);
        if (e.Cost < 0.f) Heap.emplace(e.Cost, edge);
    }
    bool Exists(uint64_t key) {
        if (Current.contains(key)) return true;
        const auto found = Original.find(key);
        return found == Original.end() ? Canonical.Get(key).Count != 0u : Edges[found->second].External != 0u;
    }
    void Rotate(uint32_t edge) {
        auto &e = Edges[edge];
        const uint32_t h = e.Nodes[0], g = e.Nodes[1], hb = Next(h), hc = Prev(h), ga = Next(g), gd = Prev(g);
        const auto new_key = MeshEdgeUsers::Key(Vertex(hc), Vertex(gd));
        if (Exists(new_key)) return;
        const bool same = Vertex(h) == Vertex(g);
        const uint32_t a = Nodes[same ? g : ga].Source, b = Nodes[hb].Source, c = Nodes[hc].Source, d = Nodes[gd].Source;
        const std::array nodes{h, hb, hc, g, ga, gd};
        std::array<uint32_t, 6> old_edges;
        for (uint32_t i = 0u; i < 6u; ++i) {
            old_edges[i] = Nodes[nodes[i]].Edge;
            for (auto &user : Edges[old_edges[i]].Nodes)
                if (user == nodes[i]) user = InvalidOffset;
        }
        Current.erase(Key(h));
        Current.emplace(new_key, edge);
        Nodes[h].Source = c;
        Nodes[hb].Source = d;
        Nodes[hc].Source = b;
        Nodes[g].Source = d;
        Nodes[ga].Source = same ? a : c;
        Nodes[gd].Source = same ? c : a;
        for (const auto node : nodes) {
            auto &at = Nodes[node].Edge;
            at = Current.at(Key(node));
            auto &users = Edges[at].Nodes;
            if (users[0] == InvalidOffset) users[0] = node;
            else if (users[1] == InvalidOffset) users[1] = node;
        }
        e.States.insert(State(edge, false));
        ++Rotations;
        for (const auto other : old_edges)
            if (other != edge) Update(other);
    }
};
} // namespace

std::optional<MeshTopologyTask> BeautifyFaceTask(const MeshStore &store, const Mesh &mesh, bool angle) {
    Planner p{mesh, angle};
    const auto selected_edges = store.GetSelectedElements(mesh.GetStoreId(), Element::Edge);
    store.GetSelectedElements(mesh.GetStoreId(), Element::Face).ForEach([&](uint32_t f) {
        if (mesh.GetValence(he::FH{f}) != 3u) return;
        p.Faces.push_back(f);
        for (const auto h : mesh.fh_range(he::FH{f})) {
            const auto incoming = mesh.GetConnectivity().Next(h);
            const auto key = p.Canonical.Key(incoming);
            const auto [it, inserted] = p.Current.try_emplace(key, uint32_t(p.Edges.size()));
            if (inserted) {
                p.Original.emplace(key, it->second);
                p.Edges.push_back({.Source = *incoming});
            }
            const auto edge = it->second;
            auto &e = p.Edges[edge];
            if (e.Count < 2u) e.Nodes[e.Count] = uint32_t(p.Nodes.size());
            ++e.Count;
            p.Nodes.push_back({*h, edge});
        }
    });
    for (uint32_t i = 0u; i < p.Edges.size(); ++i) {
        auto &e = p.Edges[i];
        const auto all = p.Canonical.Get(he::HH{e.Source}).Count;
        e.External = all > e.Count ? all - e.Count : 0u;
        e.Eligible = all == 2u && e.Count == 2u && selected_edges.Contains(*mesh.GetEdge(he::HH{e.Source}));
        if (e.Eligible) p.Update(i);
    }
    while (!p.Heap.empty()) {
        const auto edge = p.Heap.begin()->second;
        p.Heap.erase(p.Heap.begin());
        p.Edges[edge].Cost = NoCost;
        p.Rotate(edge);
    }
    profile::RecordCounter("BeautifyRotations", p.Rotations);
    if (!p.Rotations) return {};
    MeshTopologyTask task{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::ReplaceFaces, .List = {0u, uint32_t(p.Faces.size())}};
    task.List.resize(2u + p.Faces.size());
    for (uint32_t f = 0u; f < p.Faces.size(); ++f) {
        task.List[2u + f] = uint32_t(task.List.size());
        task.List.insert(task.List.end(), {p.Faces[f], 1u, 3u});
        for (uint32_t i = 0u; i < 3u; ++i)
            task.List.insert(task.List.end(), {p.Nodes[3u * f + i].Source, p.Edges[p.Nodes[Prev(3u * f + i)].Edge].Source});
    }
    return task;
}
