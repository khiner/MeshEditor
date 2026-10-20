#pragma once

#include "render/RenderSchema.h"
#include "state/DirtySet.h"

// Render instance, material, lighting, and environment change sets, bound by RegisterSceneComponentHandlers.
struct RenderChanges {
    state::DirtySet InstanceVisibility; // An Instance or Hidden change that the settle pass derives RenderInstance from.
    state::DirtySet InstanceCreated;
    state::DirtySet InstanceDestroyed; // A live entity lost its render instance.
    state::DirtySet Materials; // Material values changed in the UMA material buffer.
    state::DirtySet PbrMeshFeatures; // A mesh's PBR features changed.
    state::DirtySet ActiveMaterialVariant, MaterializedTextures, SceneWorld, PunctualLight;
    state::DirtySet MorphWeights; // Morph weights changed in the UMA weight buffer.
    state::DirtySet SceneLighting; // An image light changed.
    state::DirtySet CameraLens; // A camera object's lens changed.
};
