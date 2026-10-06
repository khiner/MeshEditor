#include "mesh/SnapSymmetry.h"
#include "Profile.h"
#include "mesh/MeshStore.h"
#include <algorithm>
#include <array>
#include <unordered_set>

std::vector<SymmetrySnapVertex> PlanSymmetrySnap(const MeshStore &meshes, const Mesh &mesh, uint32_t axis, float threshold, bool center) {
    const profile::CpuScope scope{"PlanSymmetrySnap"};
    const auto selected = meshes.GetSelectedElements(mesh.GetStoreId(), Element::Vertex);
    const auto &tree = selected.Tree;
    if (!selected.Count() || tree.Root == InvalidOffset) return {};
    const auto &a = meshes.Arenas();
    const auto aggregates = a.SelectionTree.Aggregates.Buffer.GetSpan<SelectionAggregate>();
    const auto membership = a.Vertices.Blocks.Buffer.GetSpan<MeshElementBlock>();
    std::unordered_set<uint32_t> used;
    std::vector<SymmetrySnapVertex> output;
    uint64_t visited_nodes = 0u, visited_vertices = 0u;
    selected.ForEach([&](uint32_t v) {
        if (used.contains(v)) return;
        auto mirrored = mesh.GetPosition(he::VH{v});
        mirrored[axis] = -mirrored[axis];
        float nearest = threshold * threshold;
        uint32_t partner = InvalidOffset;
        const auto distance = [&](const AABB &bounds) {
            const auto delta = Max(Max(bounds.Min - mirrored, mirrored - bounds.Max), vec3{});
            return Dot(delta, delta);
        };
        const auto walk = [&](auto &&self, uint32_t id, uint32_t level) -> void {
            if (distance(aggregates[id].Bounds) > nearest) return;
            ++visited_nodes;
            struct Child {
                uint32_t Id;
                float Distance;
            };
            std::array<Child, 16> children;
            uint32_t count = 0u;
            for (const auto child : tree.Nodes[id].Children) {
                if (child == InvalidOffset) continue;
                const auto &bounds = level ? aggregates[child].Bounds : tree.Leaves[child].Bounds;
                const float d = distance(bounds);
                if (d <= nearest) children[count++] = {child, d};
            }
            std::sort(children.begin(), children.begin() + count, [](const auto &x, const auto &y) { return x.Distance < y.Distance; });
            for (uint32_t i = 0u; i < count; ++i) {
                const auto [child, d] = children[i];
                if (d > nearest) break;
                if (level) {
                    self(self, child, level - 1u);
                    continue;
                }
                const auto &block = membership[child];
                for (uint32_t w = 0u; w < MeshElementBlockWords; ++w)
                    for (auto bits = block.Live[w]; bits; bits &= bits - 1u) {
                        const uint32_t other = child * MeshElementBlockSize + w * 32u + uint32_t(std::countr_zero(bits));
                        ++visited_vertices;
                        const auto delta = mesh.GetPosition(he::VH{other}) - mirrored;
                        const float squared = Dot(delta, delta);
                        if (squared < nearest || (partner != InvalidOffset && squared == nearest && other < partner)) {
                            nearest = squared;
                            partner = other;
                        }
                    }
            }
        };
        walk(walk, tree.Root, SelectionIndexLevels - 1u);
        if (partner == InvalidOffset || used.contains(partner) || (partner == v && !center)) return;
        // Ambiguous neighborhoods use the first canonical selected pair. A
        // destination is written once, including coincident or overlapping pairs.
        used.insert(v);
        used.insert(partner);
        output.push_back({v, partner});
        if (partner != v) output.push_back({partner, v});
    });
    std::ranges::sort(output, {}, &SymmetrySnapVertex::Vertex);
    profile::RecordCounter("SymmetrySearchNodes", visited_nodes);
    profile::RecordCounter("SymmetrySearchVertices", visited_vertices);
    return output;
}
