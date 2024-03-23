#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "BBox.h"

// Bounding Volume Heirarchy ray intersection acceleration data structure.
struct BVH {
    struct Node;

    BVH(std::vector<BBox> &&);
    ~BVH();

    const BBox &GetBox() const;
    std::vector<BBox> CreateBoxes() const;

    std::optional<float> Intersect(const Ray &) const;

private:
    std::vector<BBox> Boxes;
    const std::unique_ptr<Node> Root;

    std::optional<float> IntersectNode(const Node *, const Ray &) const;
};
