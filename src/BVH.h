#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include "BBox.h"

// Bounding Volume Heirarchy acceleration data structure.
struct BVH {
    struct Node {
        const BBox Box;
        std::unique_ptr<Node> Left, Right;
        // Index range for child objects into `BVH.Boxes`.
        // Valid only for leaf nodes.
        uint Start{uint(-1)}, End{uint(-1)};

        // For leaf nodes
        Node(BBox &&box, uint start, uint end) : Box(std::move(box)), Start(start), End(end) {}
        // For internal nodes
        Node(BBox &&box, std::unique_ptr<Node> left, std::unique_ptr<Node> right)
            : Box(std::move(box)), Left(std::move(left)), Right(std::move(right)) {}

        bool IsLeaf() const { return Left == nullptr && Right == nullptr; }
    };

    // Higher values means bigger bounding boxes with more objects inside them.
    inline static const uint MaxLeafNodeObjectCount = 4;

    std::vector<BBox> Boxes;
    const Node Root;

    BVH(std::vector<BBox> &&boxes) : Boxes(std::move(boxes)), Root(Build(0, Boxes.size())) {}

    const BBox &GetBox() { return Root.Box; }
    std::optional<float> Intersect(const Ray &ray) { return IntersectNode(&Root, ray); }

private:
    Node Build(uint start, uint end) {
        BBox bbox = BBox::UnionAll(std::span(Boxes.begin() + start, Boxes.begin() + end));
        if (end - start <= MaxLeafNodeObjectCount) return {std::move(bbox), start, end};

        // Sort this range of objects (in place) based on their center along the split axis.
        const uint split_axis = bbox.MaxAxis();
        std::sort(Boxes.begin() + start, Boxes.begin() + end, [split_axis](const auto &a, const auto &b) {
            return a.Center()[split_axis] < b.Center()[split_axis];
        });

        const uint mid = start + (end - start) / 2;
        return {std::move(bbox), std::make_unique<Node>(Build(start, mid)), std::make_unique<Node>(Build(mid, end))};
    }

    std::optional<float> IntersectNode(const Node *node, const Ray &ray) const {
        if (node == nullptr || !node->Box.Intersect(ray)) return {};

        if (node->IsLeaf()) {
            std::optional<float> min_dist = {};
            for (uint i = node->Start; i < node->End; ++i) {
                auto hit_dist = Boxes[i].Intersect(ray);
                if (hit_dist && (!min_dist || *hit_dist < *min_dist)) min_dist = hit_dist;
            }
            return min_dist;
        }

        // Recurse children.
        const std::optional<float> left_hit = IntersectNode(node->Left.get(), ray), right_hit = IntersectNode(node->Right.get(), ray);
        if (left_hit && right_hit) return *left_hit < *right_hit ? left_hit : right_hit;

        return left_hit ? left_hit : right_hit;
    }
};
