#pragma once
#include "mesh/Mesh.h"
#include <unordered_map>
#include <unordered_set>

// Sparse canonical face incidence, including same-direction and nonmanifold users.
// Each lower endpoint's fan is inspected once; no positions are copied.
class MeshEdgeUsers {
public:
    struct Users {
        uint32_t First{InvalidOffset}, Second{InvalidOffset}, Count{};
    };
    explicit MeshEdgeUsers(const Mesh &mesh) : M{mesh} {}
    static uint64_t Key(uint32_t a, uint32_t b) { return uint64_t(std::min(a, b)) << 32u | std::max(a, b); }
    uint64_t Key(he::HH h) const { return Key(*M.GetFromVertex(h), *M.GetToVertex(h)); }
    Users Get(he::HH h) { return Get(Key(h)); }
    // Visit all face sides once at their lower endpoint. Callers can retain
    // either a compact incidence summary or full radial users without copying geometry.
    void ForEachAtVertex(uint32_t v, auto &&visit) const {
        const auto &c = M.GetConnectivity();
        const auto fan = c.VertexCorners[v];
        for (uint32_t i = 0u; i < fan.y; ++i) {
            const he::HH corner{c.FanItems[fan.x + i].x};
            if (!c.FaceOf(corner)) continue;
            for (const auto side : {corner, c.Next(corner)}) {
                const auto edge = Key(side);
                if (uint32_t(edge >> 32u) == v && uint32_t(edge) != v) visit(edge, side);
            }
        }
    }
    Users Get(uint64_t edge) {
        const uint32_t v = uint32_t(edge >> 32u);
        if (Scanned.insert(v).second) ForEachAtVertex(v, [&](uint64_t key, he::HH side) {
            auto &users = Edges[key];
            if (!users.Count) users.First = *side;
            else if (users.Count == 1u) users.Second = *side;
            ++users.Count;
        });
        const auto found = Edges.find(edge);
        return found == Edges.end() ? Users{} : found->second;
    }

private:
    const Mesh &M;
    std::unordered_set<uint32_t> Scanned;
    std::unordered_map<uint64_t, Users> Edges;
};
