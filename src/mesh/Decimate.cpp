#include "mesh/Decimate.h"
#include "Profile.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>

namespace {
// Symmetric plane quadric, accumulated in double precision on the host.
struct Quadric {
    std::array<double, 10> Q{};
    void Add(vec3 n, double d, double weight = 1.) {
        const std::array<double, 4> p{n.x, n.y, n.z, d};
        for (uint32_t i = 0u, k = 0u; i < 4u; ++i)
            for (uint32_t j = i; j < 4u; ++j) Q[k++] += weight * p[i] * p[j];
    }
    Quadric operator+(const Quadric &b) const {
        Quadric q = *this;
        for (uint32_t i = 0u; i < 10u; ++i) q.Q[i] += b.Q[i];
        return q;
    }
    double Cost(vec3 p) const {
        const double x = p.x, y = p.y, z = p.z;
        return std::max(0., Q[0] * x * x + 2. * Q[1] * x * y + 2. * Q[2] * x * z + 2. * Q[3] * x + Q[4] * y * y + 2. * Q[5] * y * z + 2. * Q[6] * y + Q[7] * z * z + 2. * Q[8] * z + Q[9]);
    }
    vec3 Minimum(vec3 a, vec3 b) const {
        vec3 best = a;
        for (const auto p : {b, (a + b) * .5f})
            if (Cost(p) < Cost(best)) best = p;
        double m[3][4] = {{Q[0], Q[1], Q[2], -Q[3]}, {Q[1], Q[4], Q[5], -Q[6]}, {Q[2], Q[5], Q[7], -Q[8]}};
        const double scale = std::max({std::abs(Q[0]), std::abs(Q[4]), std::abs(Q[7])});
        for (uint32_t i = 0u; i < 3u; ++i) {
            uint32_t pivot = i;
            for (uint32_t j = i + 1u; j < 3u; ++j)
                if (std::abs(m[j][i]) > std::abs(m[pivot][i])) pivot = j;
            if (std::abs(m[pivot][i]) <= 1e-10 * scale) return best;
            for (uint32_t k = i; k < 4u; ++k) std::swap(m[pivot][k], m[i][k]);
            const double divisor = m[i][i];
            for (uint32_t k = i; k < 4u; ++k) m[i][k] /= divisor;
            for (uint32_t j = 0u; j < 3u; ++j)
                if (i != j) {
                    const double factor = m[j][i];
                    for (uint32_t k = i; k < 4u; ++k) m[j][k] -= factor * m[i][k];
                }
        }
        const vec3 p{float(m[0][3]), float(m[1][3]), float(m[2][3])};
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) && Cost(p) < Cost(best)) best = p;
        return best;
    }
};

struct Planner {
    const Mesh &M;
    const MeshStore &Store;
    struct Vertex {
        uint32_t Handle, Parent, Version{};
        bool Selected{}, Locked{}, Boundary{};
        Quadric Error;
        std::optional<vec3> Replacement;
        std::set<uint32_t> Faces;
    };
    struct Face {
        uint32_t Handle;
        std::vector<uint32_t> Vertices;
    };
    struct Candidate {
        double Error;
        uint32_t A, B, VersionA, VersionB;
        vec3 Position;
        bool operator>(const Candidate &b) const { return std::tie(Error, A, B) > std::tie(b.Error, b.A, b.B); }
    };
    std::vector<Vertex> Vertices;
    std::vector<Face> Faces;
    std::unordered_map<uint32_t, uint32_t> Index;
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> Heap;
    uint32_t Triangles{}, Collapses{};

    uint32_t AddVertex(uint32_t handle) {
        const auto [it, inserted] = Index.try_emplace(handle, uint32_t(Vertices.size()));
        if (inserted) Vertices.push_back({.Handle = handle, .Parent = it->second});
        return it->second;
    }
    vec3 Position(uint32_t v) const { return Vertices[v].Replacement.value_or(M.GetPosition(he::VH{Vertices[v].Handle})); }
    uint32_t Root(uint32_t v) const {
        while (Vertices[v].Parent != v) v = Vertices[v].Parent;
        return v;
    }
    std::set<uint32_t> Neighbors(uint32_t v) const {
        std::set<uint32_t> result;
        for (const auto f : Vertices[v].Faces) {
            const auto &loop = Faces[f].Vertices;
            for (uint32_t i = 0u; i < loop.size(); ++i)
                if (loop[i] == v) {
                    result.insert(loop[(i + 1u) % loop.size()]);
                    result.insert(loop[(i + loop.size() - 1u) % loop.size()]);
                }
        }
        return result;
    }
    std::set<uint32_t> Affected(uint32_t a, uint32_t b) const {
        auto faces = Vertices[a].Faces;
        faces.insert(Vertices[b].Faces.begin(), Vertices[b].Faces.end());
        return faces;
    }
    std::vector<uint32_t> Replaced(const Face &f, uint32_t a, uint32_t b) const {
        std::vector<uint32_t> loop;
        for (auto v : f.Vertices) {
            if (v == b) v = a;
            if (loop.empty() || loop.back() != v) loop.push_back(v);
        }
        if (loop.size() > 1u && loop.front() == loop.back()) loop.pop_back();
        return loop;
    }
    vec3 Normal(const std::vector<uint32_t> &loop, uint32_t moved = InvalidOffset, vec3 at = {}) const {
        vec3 n{};
        if (loop.empty()) return n;
        const auto p = [&](uint32_t v) { return v == moved ? at : Position(v); };
        const auto origin = p(loop.front());
        for (uint32_t i = 1u; i + 1u < loop.size(); ++i) n += Cross(p(loop[i]) - origin, p(loop[i + 1u]) - origin);
        return n;
    }
    bool Valid(uint32_t a, uint32_t b, vec3 position) const {
        const auto an = Neighbors(a), bn = Neighbors(b);
        if (!an.contains(b)) return false;
        std::set<uint32_t> common, opposites;
        std::set_intersection(an.begin(), an.end(), bn.begin(), bn.end(), std::inserter(common, common.end()));
        uint32_t users = 0u;
        for (const auto f : Vertices[a].Faces)
            if (Vertices[b].Faces.contains(f)) {
                const auto &loop = Faces[f].Vertices;
                for (uint32_t i = 0u; i < loop.size(); ++i)
                    if ((loop[i] == a && loop[(i + 1u) % loop.size()] == b) || (loop[i] == b && loop[(i + 1u) % loop.size()] == a)) ++users;
                if (loop.size() == 3u)
                    for (const auto v : loop)
                        if (v != a && v != b) opposites.insert(v);
            }
        if (users < 1u || users > 2u || common != opposites) return false;
        // Boundary vertices only collapse along the boundary itself.
        if (Vertices[a].Boundary != Vertices[b].Boundary || (Vertices[a].Boundary && users != 1u)) return false;
        std::set<std::vector<uint32_t>> unique;
        for (const auto f : Affected(a, b)) {
            auto loop = Replaced(Faces[f], a, b);
            if (loop.size() < 3u) continue;
            const auto before = Normal(Faces[f].Vertices), after = Normal(loop, a, position);
            if (!(Dot(before, after) > 1e-6f * Length(before) * Length(after)) || !(Length(after) > 1e-8f * Length(before))) return false;
            std::ranges::sort(loop);
            if (std::adjacent_find(loop.begin(), loop.end()) != loop.end() || !unique.insert(std::move(loop)).second) return false;
        }
        return !unique.empty();
    }
    void Propose(uint32_t a, uint32_t b) {
        if (a > b) std::swap(a, b);
        const auto &va = Vertices[a], &vb = Vertices[b];
        if (a == b || va.Parent != a || vb.Parent != b || !va.Selected || !vb.Selected || va.Locked || vb.Locked) return;
        const auto q = va.Error + vb.Error;
        const auto p = q.Minimum(Position(a), Position(b));
        Heap.push({q.Cost(p), a, b, va.Version, vb.Version, p});
    }
    void Collapse(const Candidate &c) {
        const auto affected = Affected(c.A, c.B);
        std::set<uint32_t> changed;
        for (const auto f : affected) {
            auto &face = Faces[f];
            Triangles -= uint32_t(face.Vertices.size()) - 2u;
            for (const auto v : face.Vertices) {
                Vertices[v].Faces.erase(f);
                changed.insert(v);
            }
            face.Vertices = Replaced(face, c.A, c.B);
            if (face.Vertices.size() < 3u) face.Vertices.clear();
            else {
                Triangles += uint32_t(face.Vertices.size()) - 2u;
                for (const auto v : face.Vertices) Vertices[v].Faces.insert(f);
            }
        }
        Vertices[c.A].Replacement = c.Position;
        Vertices[c.A].Error = Vertices[c.A].Error + Vertices[c.B].Error;
        Vertices[c.B].Parent = c.A;
        ++Collapses;
        for (const auto v : changed) ++Vertices[v].Version;
        std::set<std::pair<uint32_t, uint32_t>> edges;
        for (const auto v : changed)
            if (Root(v) == v)
                for (const auto other : Neighbors(v)) edges.emplace(std::min(v, other), std::max(v, other));
        for (const auto [a, b] : edges) Propose(a, b);
    }
    void Gather() {
        const auto &a = Store.Arenas();
        const auto &record = Store.Get(M.GetStoreId());
        const auto &connectivity = M.GetConnectivity();
        std::set<uint32_t> faces;
        Store.GetSelectedElements(M.GetStoreId(), Element::Vertex).ForEach([&](uint32_t handle) {
            Vertices[AddVertex(handle)].Selected = true;
            const auto fan = connectivity.VertexCorners[handle];
            for (uint32_t i = 0u; i < fan.y; ++i) faces.insert(connectivity.FanItems[fan.x + i].y);
        });
        const auto hidden = Store.GetHiddenElements(M.GetStoreId(), Element::Face);
        for (const auto face : faces) {
            Face f{.Handle = face};
            for (const auto v : M.fv_range(he::FH{face})) f.Vertices.push_back(AddVertex(*v));
            if (f.Vertices.size() < 3u) continue;
            const auto n = M.GetNormal(he::FH{face});
            const double d = -Dot(n, Position(f.Vertices.front()));
            for (const auto v : f.Vertices) {
                Vertices[v].Faces.insert(uint32_t(Faces.size()));
                Vertices[v].Error.Add(n, d);
                Vertices[v].Locked |= hidden.Contains(face);
            }
            Triangles += uint32_t(f.Vertices.size()) - 2u;
            Faces.push_back(std::move(f));
        }
        struct Side {
            uint32_t Face, Corner;
        };
        std::map<std::pair<uint32_t, uint32_t>, std::vector<Side>> edges;
        for (uint32_t f = 0u; f < Faces.size(); ++f) {
            const auto &loop = Faces[f].Vertices;
            for (uint32_t i = 0u; i < loop.size(); ++i) {
                const auto v = loop[i], w = loop[(i + 1u) % loop.size()];
                edges[{std::min(v, w), std::max(v, w)}].push_back({f, *connectivity.FaceHalfedge(Faces[f].Handle) + i});
            }
        }
        for (const auto &[edge, sides] : edges) {
            const auto [v, w] = edge;
            bool locked = sides.size() > 2u;
            for (const auto side : sides) locked |= a.EdgeSharpness.Get({*M.GetEdge(connectivity.Next(he::HH{side.Corner})), 1u})[0] != 0u;
            if (sides.size() == 2u) {
                const auto &f = Faces[sides[0].Face], &g = Faces[sides[1].Face];
                locked |= a.FacePrimitives.Get(f.Handle) != a.FacePrimitives.Get(g.Handle);
                for (const auto vertex : {v, w}) {
                    const auto corner = [&](Side side) { const he::HH h{side.Corner}; return *M.GetToVertex(h) == Vertices[vertex].Handle ? *h : *connectivity.Next(h); };
                    const auto h = corner(sides[0]), k = corner(sides[1]);
                    for (uint32_t uv = 0u; uv < 4u; ++uv)
                        if (record.CornerAttributes & (MeshAttributeBit_TexCoord0 << uv)) locked |= a.CornerUvs[uv].Get(h) != a.CornerUvs[uv].Get(k);
                    if (record.CornerAttributes & MeshAttributeBit_Color0) locked |= a.CornerColors.Get(h) != a.CornerColors.Get(k);
                    if (record.CornerAttributes & MeshAttributeBit_Tangent) locked |= a.CornerTangents.Get(h) != a.CornerTangents.Get(k);
                }
            }
            if (sides.size() == 1u) {
                Vertices[v].Boundary = Vertices[w].Boundary = true;
                const auto direction = Position(w) - Position(v);
                const auto normal = Cross(direction, M.GetNormal(he::FH{Faces[sides[0].Face].Handle}));
                const float length = Length(normal);
                if (length > 0.f) {
                    const auto n = normal / length;
                    const double d = -Dot(n, Position(v));
                    Vertices[v].Error.Add(n, d, 100.);
                    Vertices[w].Error.Add(n, d, 100.);
                }
            }
            if (locked) Vertices[v].Locked = Vertices[w].Locked = true;
        }
        for (const auto &[edge, sides] : edges) Propose(edge.first, edge.second);
    }
};
} // namespace

std::optional<MeshTopologyTask> DecimateTask(const MeshStore &store, const Mesh &mesh, float ratio) {
    const profile::CpuScope scope{"PlanDecimate"};
    if (!std::isfinite(ratio) || ratio >= 1.f || !mesh.FaceCount()) return {};
    Planner plan{mesh, store};
    plan.Gather();
    const auto target = uint32_t(std::ceil(double(plan.Triangles) * std::max(0.f, ratio)));
    while (plan.Triangles > target && !plan.Heap.empty()) {
        const auto candidate = plan.Heap.top();
        plan.Heap.pop();
        if (plan.Vertices[candidate.A].Version != candidate.VersionA || plan.Vertices[candidate.B].Version != candidate.VersionB) continue;
        // Validate only the next collapse, not every neighboring proposal.
        if (!plan.Valid(candidate.A, candidate.B, candidate.Position)) continue;
        plan.Collapse(candidate);
    }
    if (!plan.Collapses) return {};
    MeshTopologyTask task{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::Decimate, .List = {0u}};
    for (uint32_t v = 0u; v < plan.Vertices.size(); ++v) {
        const auto root = plan.Root(v);
        if (v == root && !plan.Vertices[v].Replacement) continue;
        const auto position = plan.Position(root);
        task.List.insert(task.List.end(), {plan.Vertices[v].Handle, plan.Vertices[root].Handle, std::bit_cast<uint32_t>(position.x), std::bit_cast<uint32_t>(position.y), std::bit_cast<uint32_t>(position.z)});
        ++task.List[0];
    }
    profile::RecordCounter("DecimateSourceFaces", plan.Faces.size());
    profile::RecordCounter("DecimateCollapsedEdges", plan.Collapses);
    return task;
}
