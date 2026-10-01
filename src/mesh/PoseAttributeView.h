#pragma once
#include "gpu/PoseAttributeNode.h"
#include <cassert>
#include <span>

template<typename T>
struct PoseAttributeView {
    std::span<const PoseAttributeNode> Nodes;
    std::span<const T> Values;
    uint32_t Root{InvalidOffset};
    bool empty() const { return Root == InvalidOffset; }
    uint32_t Find(uint32_t record) const {
        if (Root == InvalidOffset) return InvalidOffset;
        const auto child = Nodes[Root].Children[record >> (8u + PoseAttributeRadixBits)];
        if (!child) return InvalidOffset;
        const auto value = Nodes[child - 1u].Children[(record >> 8u) & PoseAttributeRadixMask];
        return value ? (value - 1u) * 256u + (record & 255u) : InvalidOffset;
    }
    uint32_t Index(uint32_t record) const { const auto index = Find(record); assert(index != InvalidOffset); return index; }
    const T &operator[](uint32_t record) const { return Values[Index(record)]; }
};
