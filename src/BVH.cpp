#include "BVH.h"

// Higher values means bigger bounding boxes with more objects inside them.
constexpr uint MaxLeafNodeObjectCount = 4;

struct BVH::Node {
    const BBox Box;
    // todo for a more flat memory structure, use a single array of nodes and store left/right indices instead of node pointers.
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

// todo build bottom-up instead of top-down for better performance.
BVH::Node Build(std::vector<BBox> &boxes, uint start, uint end) {
    BBox box = BBox::UnionAll(std::span(boxes.begin() + start, boxes.begin() + end));
    if (end - start <= MaxLeafNodeObjectCount) return {std::move(box), start, end};

    // Sort this range of objects (in place) based on their center along the split axis.
    const uint split_axis = box.MaxAxis();
    std::sort(boxes.begin() + start, boxes.begin() + end, [split_axis](const auto &a, const auto &b) {
        return a.Center()[split_axis] < b.Center()[split_axis];
    });

    const uint mid = start + (end - start) / 2;
    return {std::move(box), std::make_unique<BVH::Node>(Build(boxes, start, mid)), std::make_unique<BVH::Node>(Build(boxes, mid, end))};
}

BVH::BVH(std::vector<BBox> &&boxes) : Boxes(std::move(boxes)), Root(std::make_unique<Node>(Build(Boxes, 0, Boxes.size()))) {}
BVH::~BVH() = default;

const BBox &BVH::GetBox() const { return Root->Box; }

void AddNodeBoxes(const BVH::Node *node, std::vector<BBox> &boxes) {
    if (node == nullptr) return;

    // Add the current node's box.
    boxes.push_back(node->Box);

    // Recursively add boxes from the left and right children.
    AddNodeBoxes(node->Left.get(), boxes);
    AddNodeBoxes(node->Right.get(), boxes);
}

std::vector<BBox> BVH::CreateBoxes() const {
    std::vector<BBox> boxes;
    AddNodeBoxes(Root.get(), boxes);
    return boxes;
}

std::optional<float> BVH::Intersect(const Ray &ray) const { return IntersectNode(Root.get(), ray); }

std::optional<float> BVH::IntersectNode(const Node *node, const Ray &ray) const {
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
