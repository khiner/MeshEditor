#pragma once

#include "state/Slot.h"

struct GpuBuffers;
struct PosedLayout;
struct MeshEditWork;
struct MeshVertexOverlays;
struct BoundsRun;
struct TalliedRecord;
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
struct SceneLight;
struct ImageLight;
struct Material;
struct MeshMaterialAssignment;
struct MeshMaterialSlotSelection;
struct MaterialVariants;
struct PbrMeshFeatures;
struct Pipelines;
struct RenderChanges;
struct ActiveSamplerAnisotropy;
struct TextureStore;
struct EnvironmentStore;
struct MaterializedTexture;
struct ViewportRenderResources;

using RenderSchema = state::Schema<state::RenderBase, state::RenderCount, GpuBuffers, PosedLayout, MeshEditWork, MeshVertexOverlays, BoundsRun, TalliedRecord, MeshletPositionsDirty, ClusterLodDirty, ClusterLodDemand, MeshDisplayDirty, MeshLayoutDirty, MeshBoundsDirty, GpuSceneState, Instance, Hidden, Visibility, SceneLight, ImageLight, Material, MeshMaterialAssignment, MeshMaterialSlotSelection, MaterialVariants, PbrMeshFeatures, Pipelines, RenderChanges, ActiveSamplerAnisotropy, TextureStore, EnvironmentStore, MaterializedTexture, ViewportRenderResources>;
template<typename T>
    requires RenderSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = RenderSchema::Of<T>();
