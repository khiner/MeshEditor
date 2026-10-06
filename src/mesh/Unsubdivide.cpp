#include "mesh/Unsubdivide.h"
#include "Profile.h"
#include "mesh/Mesh.h"
#include "mesh/MeshEdgeUsers.h"
#include <algorithm>
#include <map>
#include <set>

std::optional<MeshTopologyTask> UnsubdivideTask(const Mesh &mesh, const GeometrySelection &selection) {
    mesh.ValidateSelection(selection);
    const profile::CpuScope scope{"PlanUnsubdivide"};
    const auto &c = mesh.GetConnectivity();
    const auto incidence = mesh.GetVertexEdgeIncidence();
    MeshEdgeUsers users{mesh};
    struct Node {
        uint32_t Vertex;
        std::vector<uint32_t> Neighbors;
        std::span<const uvec2> Corners;
        int Tag{};
        bool Chain{};
    };
    std::vector<Node> nodes;
    std::unordered_map<uint32_t, uint32_t> index;
    std::ranges::for_each(selection.Vertices, [&](uint32_t v) {
        const auto fan = c.VertexCorners[v];
        if (fan.y > 4u) return;
        Node node{.Vertex = v};
        uint32_t boundary = 0u, manifold = 0u;
        for (const auto e : incidence.Incident(v)) {
            if (node.Neighbors.size() == 4u) return;
            const auto h = mesh.GetHalfedge(he::EH{e}, 0u);
            const auto a = *mesh.GetFromVertex(h), b = *mesh.GetToVertex(h);
            if (a == b) return;
            node.Neighbors.push_back(a == v ? b : a);
            if (mesh.FaceCount()) {
                const auto count = users.Get(h).Count;
                if (count == 1u) ++boundary;
                else if (count == 2u) ++manifold;
                else return;
            }
        }
        const auto degree = node.Neighbors.size();
        if (!mesh.FaceCount()) {
            if (degree != 2u) return;
            node.Chain = true;
        } else {
            node.Chain = degree == 2u && manifold == 2u;
            if (!node.Chain && !((degree == 3u || degree == 4u) && manifold == degree) && !(degree == 3u && boundary == 2u && manifold == 1u)) return;
            node.Corners = c.FanItems.subspan(fan.x, fan.y);
            if (node.Chain)
                for (const auto corner : node.Corners)
                    if (mesh.GetValence(he::FH{corner.y}) <= 3u) return;
        }
        index.emplace(v, uint32_t(nodes.size()));
        nodes.push_back(std::move(node));
    });
    // Alternating breadth-first waves match Blender's keep/collapse tagging on
    // regular regions, without visiting unselected or ineligible components.
    std::vector<uint32_t> queue;
    for (uint32_t seed = 0u; seed < nodes.size(); ++seed) {
        if (nodes[seed].Tag) continue;
        nodes[seed].Tag = 1;
        queue.clear();
        queue.push_back(seed);
        for (uint32_t at = 0u; at < queue.size(); ++at) {
            const auto current = queue[at];
            for (const auto v : nodes[current].Neighbors)
                if (const auto found = index.find(v); found != index.end() && !nodes[found->second].Tag) {
                    nodes[found->second].Tag = -nodes[current].Tag;
                    queue.push_back(found->second);
                }
        }
    }
    struct Corner {
        uint32_t Source, Edge;
    };
    using Polygon = std::vector<Corner>;
    std::map<uint32_t, std::vector<Polygon>> output;
    std::set<uint32_t> removed;
    std::map<uint32_t, Polygon> fans;
    const auto vertex = [&](uint32_t h) { return *mesh.GetToVertex(he::HH{h}); };
    const auto key = [&](const Polygon &polygon) {
        std::vector<uint32_t> vertices;
        for (const auto corner : polygon) vertices.push_back(vertex(corner.Source));
        std::ranges::sort(vertices);
        return vertices;
    };
    const auto exists = [&](const std::vector<uint32_t> &vertices, const std::set<uint32_t> &excluded) {
        const auto fan = c.VertexCorners[vertices.front()];
        for (uint32_t i = 0u; i < fan.y; ++i) {
            const auto f = c.FanItems[fan.x + i].y;
            if (excluded.contains(f) || mesh.GetValence(he::FH{f}) != vertices.size()) continue;
            std::vector<uint32_t> other;
            for (const auto v : mesh.fv_range(he::FH{f})) other.push_back(*v);
            std::ranges::sort(other);
            if (vertices == other) return true;
        }
        return false;
    };
    for (const auto &node : nodes) {
        if (node.Tag != -1) continue;
        // Odd cycles can put adjacent vertices in the same wave. Keep these
        // independent so every emitted corner refers to a surviving vertex.
        if (std::ranges::any_of(node.Neighbors, [&](uint32_t v) { return removed.contains(v); })) continue;
        if (!mesh.FaceCount() || node.Chain) {
            removed.insert(node.Vertex);
            continue;
        }
        struct Ear {
            uint32_t To, Source, Edge;
        };
        std::map<uint32_t, Ear> ears;
        std::set<uint32_t> incoming, incident_faces;
        bool valid = true;
        for (const auto item : node.Corners) {
            const he::HH h{item.x};
            const auto prev = c.Previous(h), next = c.Next(h);
            const uint32_t from = vertex(*next), to = vertex(*prev);
            valid &= from != to && ears.emplace(from, Ear{to, *prev, mesh.GetValence(he::FH{item.y}) == 3u ? *prev : *h}).second && incoming.insert(to).second;
            incident_faces.insert(item.y);
        }
        if (!valid || ears.empty()) continue;
        uint32_t start = ears.begin()->first;
        bool boundary = false;
        for (const auto &[from, ear] : ears)
            if (!incoming.contains(from)) {
                start = from;
                boundary = true;
                break;
            }
        Polygon polygon;
        uint32_t at = start;
        do {
            const auto found = ears.find(at);
            if (found == ears.end()) break;
            const auto &ear = found->second;
            polygon.push_back({ear.Source, ear.Edge});
            at = ear.To;
        } while (at != start && polygon.size() <= ears.size());
        if (polygon.size() != ears.size() || (!boundary && at != start)) continue;
        if (boundary) {
            if (users.Get(MeshEdgeUsers::Key(start, at)).Count) continue;
            uint32_t source = InvalidOffset;
            for (const auto corner : node.Corners)
                if (const auto next = *c.Next(he::HH{corner.x}); vertex(next) == start) source = next;
            polygon.push_back({source, source});
        }
        if (polygon.size() < 3u || polygon.size() > 4u) continue;
        const auto vertices = key(polygon);
        if (std::adjacent_find(vertices.begin(), vertices.end()) != vertices.end() || exists(vertices, incident_faces)) continue;
        bool duplicate = false;
        std::set<std::vector<uint32_t>> residuals{vertices};
        for (const auto item : node.Corners) {
            Polygon residual;
            for (const auto corner : mesh.fh_range(he::FH{item.y}))
                if (vertex(*corner) != node.Vertex) residual.push_back({*corner, *corner});
            if (residual.size() < 3u) continue;
            const auto residual_key = key(residual);
            if (!residuals.insert(residual_key).second || exists(residual_key, incident_faces)) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) continue;
        removed.insert(node.Vertex);
        fans.emplace(node.Vertex, std::move(polygon));
    }
    if (removed.empty()) return {};
    if (!mesh.FaceCount()) {
        MeshTopologyTask task{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::DissolveVertices, .Flags = TopologyFlagListSelects, .List = {uint32_t(removed.size())}};
        task.List.insert(task.List.end(), removed.begin(), removed.end());
        task.Selection = selection;
        return task;
    }
    for (const auto v : removed)
        for (const auto corner : nodes[index.at(v)].Corners) output.try_emplace(corner.y);
    for (auto &[face, polygons] : output) {
        Polygon residual;
        for (const auto h : mesh.fh_range(he::FH{face}))
            if (!removed.contains(vertex(*h))) residual.push_back({*h, *h});
        if (residual.size() >= 3u) polygons.push_back(std::move(residual));
    }
    for (auto &[v, polygon] : fans) {
        uint32_t owner = InvalidOffset;
        for (const auto corner : nodes[index.at(v)].Corners) owner = std::min(owner, corner.y);
        output.at(owner).push_back(std::move(polygon));
    }
    std::set<std::vector<uint32_t>> unique;
    for (const auto &[face, polygons] : output)
        for (const auto &polygon : polygons)
            if (!unique.insert(key(polygon)).second) return {};
    MeshTopologyTask task{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::ReplaceFaces, .List = {uint32_t(removed.size())}};
    task.List.insert(task.List.end(), removed.begin(), removed.end());
    task.List.push_back(uint32_t(output.size()));
    const auto offsets = task.List.size();
    task.List.resize(offsets + output.size());
    uint32_t face_index = 0u;
    for (const auto &[face, polygons] : output) {
        task.List[offsets + face_index++] = uint32_t(task.List.size());
        task.List.push_back(face);
        task.List.push_back(uint32_t(polygons.size()));
        for (const auto &polygon : polygons) {
            task.List.push_back(uint32_t(polygon.size()));
            for (const auto corner : polygon) {
                task.List.push_back(corner.Source);
                task.List.push_back(corner.Edge);
            }
        }
    }
    profile::RecordCounter("UnsubdivideRemovedVertices", removed.size());
    task.Selection = selection;
    return task;
}
