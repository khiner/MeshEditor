#include "BVH.h"

#include <algorithm>
#include <numeric>
#include <ranges>

using std::ranges::to;
using std::views::filter;
using std::views::transform;

BVH::BVH(std::vector<BBox> &&leaf_boxes) : LeafBoxes(std::move(leaf_boxes)) {
    std::vector<uint> indices(LeafBoxes.size());
    std::iota(indices.begin(), indices.end(), 0);
    Build(std::move(indices));
}
BVH::~BVH() = default;

uint BVH::Build(std::vector<uint> &&indices) {
    if (indices.size() == 1) {
        Nodes.emplace_back(indices.front());
        return RootIndex();
    }

    // Partially sort the indices array around the median element,
    // based on the box centers along the longest axis of the encompassing box.
    BBox box = std::accumulate(indices.begin(), indices.end(), BBox{}, [&](const BBox &acc, uint i) {
        return acc.Union(LeafBoxes[i]);
    });
    const auto split_axis = box.MaxAxis();
    const uint mid = indices.size() / 2;
    std::nth_element(indices.begin(), indices.begin() + mid, indices.end(), [this, split_axis](auto a, auto b) {
        return LeafBoxes[a].Center()[split_axis] < LeafBoxes[b].Center()[split_axis];
    });

    // Add an internal parent node encompassing both children.
    Nodes.emplace_back(
        Build({indices.begin(), indices.begin() + mid}),
        Build({indices.begin() + mid, indices.end()}),
        std::move(box)
    );

    return RootIndex();
}

std::vector<BBox> BVH::CreateInternalBoxes() const {
    return Nodes | filter([](const auto &node) { return node.IsInternal(); }) |
        transform([](const auto &node) { return node.Internal->Box; }) | to<std::vector>();
}

std::optional<Intersection> BVH::IntersectNearest(const ray &ray, IntersectFace intersect_face, const void *userdata) const {
    if (Nodes.empty()) return {};
    Intersection nearest{0, std::numeric_limits<float>::max()};
    IntersectNode(RootIndex(), ray, intersect_face, userdata, nearest);
    if (nearest.Distance < std::numeric_limits<float>::max()) return nearest;
    return {};
}
void BVH::IntersectNode(uint node_index, const ray &ray, IntersectFace intersect_face, const void *userdata, Intersection &nearest_out) const {
    const auto &node = Nodes[node_index];
    if (auto index = node.BoxIndex) {
        if (LeafBoxes[*index].Intersect(ray)) {
            if (auto distance = intersect_face(ray, *index, userdata); distance && *distance < nearest_out.Distance) {
                nearest_out = {*index, *distance};
            }
        }
        return;
    }

    // Internal node.
    if (!node.Internal->Box.Intersect(ray)) return;
    IntersectNode(node.Internal->Left, ray, intersect_face, userdata, nearest_out);
    IntersectNode(node.Internal->Right, ray, intersect_face, userdata, nearest_out);
}
