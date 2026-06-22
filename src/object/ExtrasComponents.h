#pragma once

#include "entt_fwd.h"

#include <array>
#include <cstddef>

enum class ColliderShapeBuffer : uint8_t {
    Box,
    Sphere,
    CapsuleCap,
    Circle,
    Line,
    Count
};
struct ColliderShapeBuffers {
    std::array<entt::entity, std::size_t(ColliderShapeBuffer::Count)> Entities{
        null_entity, null_entity, null_entity, null_entity, null_entity
    };
};

struct BBoxWireframe {
    entt::entity Instance{null_entity};
};
struct TetWireframe {
    entt::entity Instance{null_entity};
};
// Wireframe overlays — rebuilt reactively from the collider/bbox/tet state they visualize.

// Marks a derived wireframe-overlay entity (a shared unit-mesh buffer or a per-collider instance), rebuilt by
// EnsureWireframes from the state it visualizes, so the snapshot skips it.
struct OverlayExtra {};
