#include "render/MeshletIndex.h"
#include <algorithm>
#include <unordered_set>
#include <vector>

namespace {
// The live bits one edit adds to and removes from one 256-handle block.
struct BlockEdit {
    uint32_t Block{};
    std::array<uint32_t,8> Added{}, Removed{};
    bool Adds() const { return std::ranges::any_of(Added,[](uint32_t word) { return word != 0u; }); }
};

// The edit's blocks in ascending order, each once.
std::vector<BlockEdit> BlockEdits(const MeshletIndexEdit &edit) {
    std::vector<BlockEdit> blocks;
    if (edit.Insert.Count) {
        const uint64_t first = edit.Insert.Offset, end = first+edit.Insert.Count;
        for (uint64_t block = first/256u; block*256u < end; ++block) {
            auto &entry = blocks.emplace_back(BlockEdit{.Block=uint32_t(block)});
            for (uint32_t w = 0u; w < 8u; ++w) {
                const uint64_t base = block*256u+w*32u, lo = std::max(base,first), hi = std::min(base+32u,end);
                entry.Added[w] = lo < hi ? (~0u >> (32u-uint32_t(hi-lo))) << uint32_t(lo-base) : 0u;
            }
        }
    }
    const auto mark = [&](std::span<const uint32_t> handles, auto words) {
        for (const auto handle : handles) {
            if (blocks.empty() || blocks.back().Block != handle/256u) blocks.push_back({.Block=handle/256u});
            (blocks.back().*words)[(handle%256u)/32u] |= 1u << (handle%32u);
        }
    };
    mark(edit.Added,&BlockEdit::Added);
    mark(edit.Removed,&BlockEdit::Removed);
    std::ranges::sort(blocks,{},&BlockEdit::Block);
    std::vector<BlockEdit> merged;
    for (const auto &entry : blocks) {
        if (merged.empty() || merged.back().Block != entry.Block) {
            merged.push_back(entry);
            continue;
        }
        for (uint32_t w = 0u; w < 8u; ++w) {
            merged.back().Added[w] |= entry.Added[w];
            merged.back().Removed[w] |= entry.Removed[w];
        }
    }
    return merged;
}
} // namespace

void MeshletIndex::Update(std::span<MeshletIndexEdit> edits) {
    std::unordered_set<uint32_t> roots;
    // Validate the whole batch before changing any ownership links.
    for (const auto &edit : edits) {
        if (edit.Root != InvalidOffset && !roots.insert(edit.Root).second) throw std::invalid_argument("Meshlet index batch repeats a root.");
        if (uint64_t(edit.Insert.Offset)+edit.Insert.Count > InvalidOffset) throw std::out_of_range("Meshlet index range exceeds canonical handles.");
        for (const auto handles : {edit.Added,edit.Removed})
            if (std::ranges::find(handles,InvalidOffset) != handles.end()) throw std::out_of_range("Meshlet index handle is invalid.");
    }
    // Fresh roots take their nodes and leaves from one run per update, a node per distinct prefix and a leaf per adding block.
    std::vector<std::vector<BlockEdit>> edit_blocks(edits.size());
    uint32_t fresh_nodes = 0u, fresh_leaves = 0u;
    for (uint32_t i = 0u; i < edits.size(); ++i) {
        edit_blocks[i] = BlockEdits(edits[i]);
        if (edits[i].Root != InvalidOffset) continue;
        std::array<uint32_t,MeshletIndexLevels> prefixes;
        prefixes.fill(InvalidOffset);
        for (const auto &entry : edit_blocks[i]) {
            if (!entry.Adds()) continue;
            ++fresh_leaves;
            for (uint32_t level = 0u; level < MeshletIndexLevels; ++level) {
                const auto prefix = entry.Block >> (5u*(level+1u));
                fresh_nodes += prefix != prefixes[level];
                prefixes[level] = prefix;
            }
        }
    }
    auto next_node = Nodes.Allocate(fresh_nodes).Offset, next_leaf = Leaves.Allocate(fresh_leaves).Offset;
    const auto new_node = [&](uint32_t parent, bool fresh) {
        const auto id = fresh ? next_node++ : Nodes.Allocate(1u).Offset;
        MeshletIndexNode node{};
        std::ranges::fill(node.Children,InvalidOffset);
        node.Parent = parent;
        Nodes.GetMutable({id,1u})[0] = node;
        return id;
    };
    // The nodes on every edited path by level, where level zero parents leaves.
    std::array<std::vector<uint32_t>,MeshletIndexLevels> paths;
    for (uint32_t i = 0u; i < edits.size(); ++i) {
        auto &edit = edits[i];
        const auto &blocks = edit_blocks[i];
        const bool fresh = edit.Root == InvalidOffset;
        if (fresh && std::ranges::any_of(blocks,&BlockEdit::Adds)) edit.Root = new_node(InvalidOffset,true);
        if (edit.Root == InvalidOffset) continue;
        for (const auto &entry : blocks) {
            std::array<uint32_t,MeshletIndexLevels> path;
            auto node = edit.Root;
            uint32_t leaf = InvalidOffset;
            for (uint32_t level = MeshletIndexLevels; level--;) {
                path[level] = node;
                const auto slot = (entry.Block >> (level*5u))&31u;
                auto child = Nodes.Get({node,1u})[0].Children[slot];
                if (child == InvalidOffset) {
                    if (!entry.Adds()) break;
                    if (level) child = new_node(node,fresh);
                    else {
                        child = fresh ? next_leaf++ : Leaves.Allocate(1u).Offset;
                        Leaves.GetMutable({child,1u})[0] = {.Parent=node,.Block=entry.Block};
                    }
                    Nodes.GetMutable({node,1u})[0].Children[slot] = child;
                }
                if (!level) leaf = child;
                else node = child;
            }
            if (leaf == InvalidOffset) continue;
            for (uint32_t level = 0u; level < MeshletIndexLevels; ++level) paths[level].push_back(path[level]);
            auto value = Leaves.Get({leaf,1u})[0];
            uint32_t first = InvalidOffset, last = 0u;
            value.Count = 0u;
            for (uint32_t w = 0u; w < 8u; ++w) {
                const auto bits = (value.Live[w] & ~entry.Removed[w]) | entry.Added[w];
                value.Live[w] = bits;
                value.Count += uint32_t(std::popcount(bits));
                if (!bits) continue;
                const auto base = entry.Block*256u+w*32u;
                first = std::min(first,base+uint32_t(std::countr_zero(bits)));
                last = base+31u-uint32_t(std::countl_zero(bits));
            }
            value.DenseFirst = value.Count && last-first+1u == value.Count ? first : InvalidOffset;
            if (value.Count) Leaves.GetMutable({leaf,1u})[0] = value;
            else {
                // Prune only paths whose edited leaves became empty.
                Nodes.GetMutable({value.Parent,1u})[0].Children[entry.Block&31u] = InvalidOffset;
                Leaves.Release({leaf,1u});
            }
        }
    }
    // Children refresh before their parents, which read their counts and dense runs.
    for (uint32_t level = 0u; level < MeshletIndexLevels; ++level) {
        auto &ids = paths[level];
        std::ranges::sort(ids);
        ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
        // A child's population and the first handle of its dense run.
        const auto run = [&](uint32_t child) {
            if (child == InvalidOffset) return std::pair{0u,InvalidOffset};
            if (level) return std::pair{Nodes.Get({child,1u})[0].Count,Nodes.Get({child,1u})[0].DenseFirst};
            return std::pair{Leaves.Get({child,1u})[0].Count,Leaves.Get({child,1u})[0].DenseFirst};
        };
        for (const auto id : ids) {
            auto node = Nodes.Get({id,1u})[0];
            uint32_t sum = 0u, active = 0u, gaps = 0u, dense_first = InvalidOffset, dense_last = 0u;
            for (uint32_t slot = 0u; slot < 32u; ++slot) {
                const auto [count,first] = run(node.Children[slot]);
                if (count) {
                    active |= 1u << slot;
                    if (first == InvalidOffset) ++gaps;
                    else {
                        dense_first = std::min(dense_first,first);
                        dense_last = std::max(dense_last,first+count-1u);
                    }
                }
                sum += count;
                node.Ends[slot] = sum;
            }
            if (!sum && node.Parent != InvalidOffset) {
                auto &parent = Nodes.GetMutable({node.Parent,1u})[0];
                *std::ranges::find(parent.Children,id) = InvalidOffset;
                Nodes.Release({id,1u});
                continue;
            }
            node.Count = sum;
            node.Active = active;
            node.DenseFirst = sum && !gaps && uint64_t(dense_last)-dense_first+1u == sum ? dense_first : InvalidOffset;
            if (std::popcount(active) == 1) {
                // A chain with one live child selects through that child's next branching node.
                const auto only = node.Children[std::countr_zero(active)];
                node.SelectNode = level ? Nodes.Get({only,1u})[0].SelectNode : only;
                node.SelectDepth = level ? Nodes.Get({only,1u})[0].SelectDepth : 0u;
            } else {
                node.SelectNode = active ? id : InvalidOffset;
                node.SelectDepth = active ? level+1u : 0u;
            }
            Nodes.GetMutable({id,1u})[0] = node;
        }
    }
}

void MeshletIndex::Release(uint32_t root) {
    Release(std::span{&root, 1u});
}
void MeshletIndex::Release(std::span<const uint32_t> roots) {
    std::vector<Range> nodes, leaves;
    Read().CollectOwned(roots, nodes, leaves, [](uint32_t, const auto &) {});
    Leaves.Release(std::move(leaves));
    Nodes.Release(std::move(nodes));
}
