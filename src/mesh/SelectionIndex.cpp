#include "mesh/SelectionIndex.h"
#include "SortUnique.h"

#include "Profile.h"
#include "numeric/VectorMath.h"

void SelectionIndex::Update(uint32_t root, std::span<const uint32_t> blocks, uint32_t owner, std::span<const MeshElementBlock> membership, std::span<const SelectionAggregate> leaves) {
    if (Roots.size() <= root) Roots.resize(size_t(root) + 1u, InvalidOffset);
    const auto allocate = [&](uint32_t parent, uint32_t slot) {
        const auto id = Aggregates.Allocate(1u).Offset;
        if (Nodes.size() <= id) Nodes.resize(size_t(id) + 1u);
        auto &node = Nodes[id];
        node = {.Parent = parent, .Slot = slot};
        node.Children.fill(InvalidOffset);
        Aggregates.GetMutable({id, 1u})[0] = {};
        return id;
    };
    if (Roots[root] == InvalidOffset) Roots[root] = allocate(InvalidOffset, 0u);
    std::array<std::vector<uint32_t>, SelectionIndexLevels> paths;
    for (const auto block : blocks) {
        const bool live = block < membership.size() && membership[block].Owner == owner && membership[block].Count;
        if (live) Owners[root % 3u][block] = root;
        else if (Owner(root % 3u, block) == root) Owners[root % 3u].erase(block);
        auto id = Roots[root];
        for (uint32_t level = SelectionIndexLevels; level--;) {
            paths[level].push_back(id);
            const auto slot = (block >> (4u * level)) & 15u;
            if (!level) {
                Nodes[id].Children[slot] = live ? block : InvalidOffset;
                break;
            }
            auto child = Nodes[id].Children[slot];
            if (child == InvalidOffset) {
                if (!live) break;
                child = allocate(id, slot);
                Nodes[id].Children[slot] = child;
            }
            id = child;
        }
    }
    // A fixed child order makes each sum a function of current values, with no
    // accumulated floating-point delta error across edits, undo, or redo.
    uint32_t updated = 0u;
    for (uint32_t level = 0u; level < SelectionIndexLevels; ++level) {
        auto &ids = paths[level];
        SortUnique(ids);
        updated += uint32_t(ids.size());
        for (const auto id : ids) {
            auto &node = Nodes[id];
            SelectionAggregate sum{};
            node.Active = node.Selected = node.Boundary = node.Hidden = node.Unselected = 0u;
            for (uint32_t slot = 0u; slot < 16u; ++slot) {
                const auto child = node.Children[slot];
                if (child == InvalidOffset) continue;
                const auto &value = level ? Aggregates.Get({child, 1u})[0] : leaves[child];
                node.Active |= 1u << slot;
                if (value.Selected) node.Selected |= 1u << slot;
                if (value.Hidden) node.Hidden |= 1u << slot;
                if (value.LiveCount > value.Selected + value.Hidden) node.Unselected |= 1u << slot;
                if (value.Flags & SelectionBoundary) node.Boundary |= 1u << slot;
                sum.PositionSum += value.PositionSum;
                sum.Bounds.Min = Min(sum.Bounds.Min, value.Bounds.Min);
                sum.Bounds.Max = Max(sum.Bounds.Max, value.Bounds.Max);
                sum.Selected += value.Selected;
                sum.Hidden += value.Hidden;
                sum.LiveCount += value.LiveCount;
                sum.Flags |= value.Flags;
            }
            if (!node.Active && node.Parent != InvalidOffset) {
                Nodes[node.Parent].Children[node.Slot] = InvalidOffset;
                Aggregates.Release({id, 1u});
            } else Aggregates.GetMutable({id, 1u})[0] = sum;
        }
    }
    profile::RecordCounter("SelectionIndexNodes", updated);
}

void SelectionIndex::Release(uint32_t root) {
    if (root >= Roots.size() || Roots[root] == InvalidOffset) return;
    const auto release = [&](auto &&self, uint32_t id, uint32_t level) -> void {
        for (const auto child : Nodes[id].Children) {
            if (child == InvalidOffset) continue;
            if (level) self(self, child, level - 1u);
            else if (Owner(root % 3u, child) == root) Owners[root % 3u].erase(child);
        }
        Aggregates.Release({id, 1u});
    };
    release(release, Roots[root], SelectionIndexLevels - 1u);
    Roots[root] = InvalidOffset;
}
