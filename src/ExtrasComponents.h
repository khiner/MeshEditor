#pragma once

#include "entt_fwd.h"

#include <array>
#include <cstddef>
#include <cstdint>

enum class ColliderShapeBuffer : uint8_t { Box,
                                           Sphere,
                                           CapsuleCap,
                                           Circle,
                                           Line,
                                           Count };
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
