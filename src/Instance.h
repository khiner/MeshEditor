#pragma once

#include "entt_fwd.h"

// Permanent linkage to the entity holding shared GPU buffers (MeshBuffers + ModelsBuffer).
// Every instance entity gets this component.
// For mesh-specific operations, check R.all_of<Mesh>(instance.Entity).
struct Instance {
    entt::entity Entity;
};
