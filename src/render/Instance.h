#pragma once

#include "state/Entity.h"

struct Instance {
    state::Entity Entity;
};

// Canonical per-object visibility, present == hidden (sparse, since most objects are visible).
// The settle pass writes it into the instance's state byte, and every draw, pick and overlay pass skips a hidden instance.
struct Hidden {};
constexpr uint8_t InstanceStateHidden{1u << 2};

// A node's own KHR_node_visibility flag. Hidden follows this flag and the flags of every ancestor.
struct Visibility {
    uint32_t Visible{1};

    bool operator==(const Visibility &) const = default;
};
