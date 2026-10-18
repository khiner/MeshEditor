#pragma once

#include "state/Slot.h"

#include <memory>

struct PendingRenderRequest;
struct WindowsState;
struct ValidationSession;
struct FrameState;
struct StartTransform;
struct StartBoneLength;
struct Interaction;
struct EditMode;
struct EnabledInteractionModes;
struct OrbitToActive;
struct ShadeSmoothAngle;
struct VideoRecording;
struct ViewCamera;
struct LookingThrough;
struct SavedViewCamera;
struct ViewportChanges;
struct ViewportConsumerFence;
struct ViewportDisplay;
struct MaterialPreviewLighting;
struct RenderedLighting;
struct StudioEnvironment;
struct ViewportExtent;
struct MeshGeometryDirty;
struct MeshPositionsChanged;
struct PendingSetEditMode;
struct PendingImportMesh;
struct ViewportIcons;
struct StartPivot;
struct PendingTransform;
struct StartScreenTransform;
struct TransformGizmoState;

using ViewportSchema = state::Schema<state::ViewportBase, state::ViewportCount, PendingRenderRequest, WindowsState, std::unique_ptr<ValidationSession>, FrameState, StartTransform, StartBoneLength, Interaction, EditMode, EnabledInteractionModes, OrbitToActive, ShadeSmoothAngle, VideoRecording, ViewCamera, LookingThrough, SavedViewCamera, ViewportChanges, ViewportConsumerFence, ViewportDisplay, MaterialPreviewLighting, RenderedLighting, StudioEnvironment, ViewportExtent, MeshGeometryDirty, MeshPositionsChanged, PendingSetEditMode, PendingImportMesh, ViewportIcons, StartPivot, PendingTransform, StartScreenTransform, TransformGizmoState>;
template<typename T>
    requires ViewportSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = ViewportSchema::Of<T>();
