#pragma once

#include "BBox.h"
#include "entt_fwd.h"

#include <array>

enum class ColliderShapeBuffer : uint8_t {
    Box,
    Sphere,
    CapsuleCap,
    Circle,
    Line,
    Count
};
struct ColliderShapeBuffers {
    std::array<entt::entity, size_t(ColliderShapeBuffer::Count)> Entities{
        null_entity, null_entity, null_entity, null_entity, null_entity
    };
};

struct BBoxWireframe {
    entt::entity Instance{null_entity};
};
// Mesh-local bounds after morph/skin deformation. Only present on deformed instances.
struct DeformedBounds {
    BBox Box;
};
struct TetWireframe {
    entt::entity Instance{null_entity};
};

// Present on a derived wireframe-overlay entity
struct OverlayExtra {};
