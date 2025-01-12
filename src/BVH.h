#pragma once

#include <optional>
#include <vector>

#include "BBox.h"
#include "Intersection.h"

struct BVH {
    struct Node {
        // Leaf nodes have a valid `BoxIndex` pointing to the index of the box in `BVH::LeafBoxes`.
        // Internal nodes have a valid `Internal` field.
        struct InternalData {
            uint Left, Right; // Indices of child nodes in `BVH::Nodes`.
            BBox Box; // Union of the child boxes.
        };
        std::optional<uint> BoxIndex;
        std::optional<InternalData> Internal;

        Node(uint index) : BoxIndex(index) {}
        Node(uint left, uint right, BBox box) : Internal({left, right, box}) {}

        bool IsLeaf() const { return BoxIndex.has_value(); }
        bool IsInternal() const { return Internal.has_value(); }
    };

    BVH(std::vector<BBox> &&leaf_boxes);
    ~BVH();

    uint RootIndex() const { return Nodes.size() - 1; }

    using IntersectFace = std::optional<float> (*)(const Ray &, uint face_index, const void *userdata);
    std::optional<Intersection> IntersectNearest(const Ray &, IntersectFace, const void *userdata) const;

    std::vector<BBox> CreateInternalBoxes() const; // All non-leaf boxes, for debugging.

private:
    std::vector<BBox> LeafBoxes;
    std::vector<Node> Nodes;

    uint Build(std::vector<uint> &&indices);
    void IntersectNode(uint node_index, const Ray &, IntersectFace, const void *userdata, Intersection &nearest_out) const;
};
