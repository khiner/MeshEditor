#pragma once

#include "state/Slot.h"

#include <memory>

struct PanelsState;
struct ValidationSession;
struct FrameState;
struct StartTransform;
struct StartBoneLength;
struct Interaction;
struct EditMode;
struct EnabledInteractionModes;
struct ShadeSmoothAngle;
struct ViewportChanges;
struct ViewportConsumerFence;
struct Viewport;
struct ViewportPanel;
struct ActiveViewport;
struct ViewCameraState;
struct OrbitToActive;
struct ViewportDisplay;
struct MaterialPreviewLighting;
struct RenderedLighting;
struct StudioEnvironment;
struct AnisotropicFilter;
struct MeshGeometryDirty;
struct MeshPositionsChanged;
struct PendingSetEditMode;
struct PendingImportMesh;
struct PendingViewSetup;
struct ViewportIcons;
struct StartPivot;
struct PendingTransform;
struct StartScreenTransform;
struct TransformGizmoState;
struct ViewportScene;

using ViewportSchema = state::Schema<state::ViewportBase, state::ViewportCount, PanelsState, std::unique_ptr<ValidationSession>, FrameState, StartTransform, StartBoneLength, Interaction, EditMode, EnabledInteractionModes, ShadeSmoothAngle, ViewportChanges, ViewportConsumerFence, AnisotropicFilter, MeshGeometryDirty, MeshPositionsChanged, PendingSetEditMode, PendingImportMesh, PendingViewSetup, ViewportIcons, StartPivot, PendingTransform, StartScreenTransform, TransformGizmoState, ViewportScene, Viewport, ViewportPanel, ActiveViewport, ViewCameraState, OrbitToActive, ViewportDisplay, MaterialPreviewLighting, RenderedLighting, StudioEnvironment>;
template<typename T>
    requires ViewportSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = ViewportSchema::Of<T>();
