#pragma once

#include "entt_fwd.h"

// Component for entities that render a mesh via instancing.
// References an entity with Mesh+MeshBuffers+ModelsBuffer components.
struct MeshInstance {
    entt::entity MeshEntity;
};
