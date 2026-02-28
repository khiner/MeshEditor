#pragma once

#include "vulkan/Buffer.h"

struct RenderInstance {
    uint32_t BufferIndex{0}; // Slot in GPU model instance buffer
    uint32_t ObjectId{0};
};

// Stored on mesh entities.
// Holds the `WorldTransform` of all instances of the mesh.
struct ModelsBuffer {
    mvk::Buffer Buffer;
    mvk::Buffer ObjectIds; // Per-instance ObjectIds for selection/silhouette rendering.
    mvk::Buffer InstanceStates; // Per-instance selection/active state flags.
};

// Cameras, lights, empties.
struct ObjectExtrasTag {};

struct VertexClass {
    uint32_t Offset{InvalidOffset};
};
