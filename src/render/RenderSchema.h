#pragma once

#include "state/Slot.h"

struct GpuBuffers;
struct PosedLayout;
struct MeshEditWork;
struct MeshVertexOverlays;
struct BoundsRun;
struct MeshletPositionsDirty;
struct ClusterLodDirty;
struct ClusterLodDemand;
struct MeshDisplayDirty;
struct MeshLayoutDirty;
struct MeshBoundsDirty;
struct GpuSceneState;
struct Instance;
struct Hidden;
struct Visibility;
struct PunctualLight;
struct ImageLight;
struct LightIndex;
struct MaterialStore;
struct MeshMaterialAssignment;
struct MeshMaterialSlotSelection;
struct MaterialVariants;
struct PbrMeshFeatures;
struct Pipelines;
struct RenderChanges;
struct ActiveSamplerAnisotropy;
struct TextureStore;
struct EnvironmentStore;
struct MaterializedTextures;
struct ViewportRenderResources;

using RenderSchema = state::Schema<state::RenderBase, state::RenderCount, GpuBuffers, PosedLayout, MeshEditWork, MeshVertexOverlays, BoundsRun, MeshletPositionsDirty, ClusterLodDirty, ClusterLodDemand, MeshDisplayDirty, MeshLayoutDirty, MeshBoundsDirty, GpuSceneState, Instance, Hidden, Visibility, PunctualLight, ImageLight, LightIndex, MaterialStore, MeshMaterialAssignment, MeshMaterialSlotSelection, MaterialVariants, PbrMeshFeatures, Pipelines, RenderChanges, ActiveSamplerAnisotropy, TextureStore, EnvironmentStore, MaterializedTextures, ViewportRenderResources>;
template<typename T>
    requires RenderSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = RenderSchema::Of<T>();
