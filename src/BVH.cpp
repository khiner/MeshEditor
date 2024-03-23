#include "BVH.h"
#include <algorithm>
#include <numeric>

BVH::BVH(std::vector<BBox> &&leaf_boxes) : LeafBoxes(std::move(leaf_boxes)) {
    std::vector<uint> indices(LeafBoxes.size());
    std::iota(indices.begin(), indices.end(), 0);
    Build(std::move(indices));
}

uint BVH::Build(std::vector<uint> &&indices) {
    const uint start = 0, end = indices.size();
    if (end - start == 1) {
        Nodes.emplace_back(indices[start]);
        return Nodes.size() - 1;
    }

    // Partially sort the indices array around the median element, based on the box centers along the longest axis.
    const uint mid = start + (end - start) / 2;
    const auto split_axis = LeafBoxes[indices[start]].MaxAxis();
    std::nth_element(indices.begin() + start, indices.begin() + mid, indices.begin() + end, [this, split_axis](auto a, auto b) {
        return LeafBoxes[a].Center()[split_axis] < LeafBoxes[b].Center()[split_axis];
    });

    // Add an internal parent node encompassing both children.
    const uint left_i = Build({indices.begin() + start, indices.begin() + mid});
    const uint right_i = Build({indices.begin() + mid, indices.begin() + end});
    const auto &left = Nodes[left_i], right = Nodes[right_i];
    const auto &left_box = left.IsLeaf() ? LeafBoxes[*left.BoxIndex] : left.Internal->Box;
    const auto &right_box = right.IsLeaf() ? LeafBoxes[*right.BoxIndex] : right.Internal->Box;
    Nodes.emplace_back(left_i, right_i, left_box.Union(right_box));

    return Nodes.size() - 1;
}

std::vector<BBox> BVH::CreateInternalBoxes() const {
    std::vector<BBox> boxes;
    boxes.reserve(Nodes.size());
    for (const auto &node : Nodes) {
        if (node.IsInternal()) boxes.emplace_back(node.Internal->Box);
    }
    return boxes;
}

std::optional<uint> BVH::Intersect(const Ray &ray, const std::function<bool(uint)> &callback) const {
    if (Nodes.empty()) return std::nullopt;

    const uint root_index = Nodes.size() - 1;
    return IntersectNode(root_index, ray, callback);
}

std::optional<uint> BVH::IntersectNode(uint node_index, const Ray &ray, const std::function<bool(uint)> &callback) const {
    const auto &node = Nodes[node_index];
    if (node.IsLeaf()) return LeafBoxes[*node.BoxIndex].Intersect(ray) && callback(*node.BoxIndex) ? node.BoxIndex : std::nullopt;

    // Internal node.
    if (!node.Internal->Box.Intersect(ray)) return std::nullopt;
    if (auto left_hit = IntersectNode(node.Internal->Left, ray, callback)) return left_hit;
    return IntersectNode(node.Internal->Right, ray, callback);
}
