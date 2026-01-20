#pragma once

namespace {
struct SelectionNode {
    float Depth;
    uint32_t ObjectId;
    uint32_t Next;
};

struct SelectionCounters {
    uint32_t Count;
    uint32_t Overflow;
};

struct ClickHit {
    float Depth;
    uint32_t ObjectId;
};

struct ClickResult {
    uint32_t Count;
    std::array<ClickHit, 64> Hits;
};

struct ClickElementCandidate {
    uint32_t ObjectId;
    float Depth;
    uint32_t DistanceSq;
};
static_assert(sizeof(ClickElementCandidate) == 12, "ClickElementCandidate must match scalar layout.");

struct ClickSelectPushConstants {
    glm::uvec2 TargetPx;
    uint32_t HeadImageIndex;
    uint32_t SelectionNodesIndex;
    uint32_t ClickResultIndex;
};

struct ClickSelectElementPushConstants {
    glm::uvec2 TargetPx;
    uint32_t Radius;
    uint32_t HeadImageIndex;
    uint32_t SelectionNodesIndex;
    uint32_t ClickResultIndex;
};

struct BoxSelectPushConstants {
    glm::uvec2 BoxMin;
    glm::uvec2 BoxMax;
    uint32_t ObjectCount;
    uint32_t HeadImageIndex;
    uint32_t SelectionNodesIndex;
    uint32_t BoxResultIndex;
};

constexpr uint32_t
    ClickSelectRadiusPx = 50,
    ClickSelectDiameterPx = ClickSelectRadiusPx * 2 + 1,
    ClickSelectPixelCount = ClickSelectDiameterPx * ClickSelectDiameterPx;
} // namespace
