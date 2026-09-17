#pragma once

#include "state/Entity.h"

struct Instance {
    state::Entity Entity;
};

// Canonical per-object visibility, present == hidden (sparse, since most objects are visible).
// RenderInstance is reactively created for an Instance without Hidden, and removed when Hidden appears.
struct Hidden {};

// A node's own KHR_node_visibility flag. Hidden follows this flag and the flags of every ancestor.
struct Visibility {
    uint32_t Visible{1};

    bool operator==(const Visibility &) const = default;
};
