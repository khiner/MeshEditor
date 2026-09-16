#pragma once

#include "gpu/DebugChannel.h"
#include "gpu/Element.h"
#include "gpu/IblSamplers.h"
#include "gpu/InteractionMode.h"
#include "gpu/Types.h"

struct SceneViewUBO {
    mat4 ViewProj DEFAULT();
    mat3 ViewRotation DEFAULT();
    vec3 CameraPosition DEFAULT();
    float CameraNear DEFAULT();
    float CameraFar DEFAULT();
    uint32_t LightCount DEFAULT();
    uint32_t LightSlot DEFAULT();
    uint32_t UseSceneLightsRender DEFAULT();
    float EnvIntensity DEFAULT();
    float Exposure DEFAULT();
    mat3 EnvRotation DEFAULT();
    float BackgroundBlur DEFAULT();
    float WorldOpacity DEFAULT();
    IblSamplers Ibl DEFAULT();
    InteractionMode InteractionMode DEFAULT();
    Element EditElement DEFAULT();
    uint32_t IsTransforming DEFAULT();
    vec3 PendingPivot DEFAULT();
    vec3 PendingTranslation DEFAULT();
    quat PendingRotation DEFAULT(1, 0, 0, 0);
    vec3 PendingScale DEFAULT();
    float ScreenPixelScale DEFAULT();
    vec2 ViewportSize DEFAULT();
    // Cluster LOD error threshold in pixels. Zero selects original geometry.
    float LodErrorPixels DEFAULT();
    uint32_t CornerTangentSlot DEFAULT();
    uint32_t CornerColorSlot DEFAULT();
    uint32_t CornerUvSlot DEFAULT();
    uint32_t EdgeSharpnessSlot DEFAULT();
    uint32_t CornerClassSlot DEFAULT();
    uint32_t CustomCornerMaskSlot DEFAULT();
    uint32_t CustomCornerNormalSlot DEFAULT();
    uint32_t BaseSeamNormalSlot DEFAULT();
    // Derived base normals indexed by draw vertex and face offsets.
    uint32_t BaseVertexNormalSlot DEFAULT();
    uint32_t BaseFaceNormalSlot DEFAULT();
    uint32_t FaceFirstTriangleSlot DEFAULT();
    uint32_t AdjacencySlot DEFAULT();
    uint32_t BoneDeformSlot DEFAULT();
    uint32_t ArmatureDeformSlot DEFAULT();
    uint32_t MorphDeformSlot DEFAULT();
    uint32_t MorphWeightsSlot DEFAULT();
    uint32_t PosedPositionSlot DEFAULT();
    uint32_t PosedVertexNormalSlot DEFAULT();
    uint32_t PosedSeamNormalSlot DEFAULT();
    uint32_t PosedFaceNormalSlot DEFAULT();
    // Weighted authored morph-normal deltas indexed by posed vertex slot.
    uint32_t PosedMorphNormalDeltaSlot DEFAULT();
    // GPU-reduced local-space instance bounds for emission-side frustum culling.
    uint32_t InstanceBoundsSlot DEFAULT(InvalidSlot);
    // Motion-blur steps use captured model transforms while preserving step-independent draw data.
    uint32_t ModelSlotOverride DEFAULT(InvalidSlot);
    uint32_t MaterialSlot DEFAULT();
    uint32_t PrimitiveMaterialSlot DEFAULT();
    uint32_t ElementPrimitiveSlot DEFAULT();
    uint32_t BoneXRay DEFAULT();
    // Opacity of every surface in the X-ray solid draw.
    float XRayAlpha DEFAULT(1);
    // Scales overlay coverage behind the scene surface while overlays draw through it. Zero leaves occlusion to the depth test.
    float OverlayBehindOpacity DEFAULT();
    uint32_t SceneDepthSamplerSlot DEFAULT(InvalidSlot);
    // Apply wire and point selection colors only when overlays are visible.
    uint32_t ShowOverlays DEFAULT();
    uint32_t ShowExtras DEFAULT();
    uint32_t ShowBoundingBoxes DEFAULT();
    uint32_t ShowTetWireframe DEFAULT();
    float NdcOffsetFactor DEFAULT();
    uint32_t TransmissionFramebufferSamplerSlot DEFAULT(InvalidSlot);
    uint32_t TransmissionFramebufferMipCount DEFAULT();
    uint32_t UseRealTransmission DEFAULT();
    DebugChannel DebugChannel DEFAULT();
};
static_assert(sizeof(SceneViewUBO) == 452, "SceneViewUBO size");
