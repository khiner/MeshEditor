#pragma once

#include "state/Slot.h"

#include <memory>

struct WindowsState;
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
struct Viewports;
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

using ViewportSchema = state::Schema<state::ViewportBase, state::ViewportCount, WindowsState, std::unique_ptr<ValidationSession>, FrameState, StartTransform, StartBoneLength, Interaction, EditMode, EnabledInteractionModes, ShadeSmoothAngle, ViewportChanges, ViewportConsumerFence, Viewports, AnisotropicFilter, MeshGeometryDirty, MeshPositionsChanged, PendingSetEditMode, PendingImportMesh, PendingViewSetup, ViewportIcons, StartPivot, PendingTransform, StartScreenTransform, TransformGizmoState>;
template<typename T>
    requires ViewportSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = ViewportSchema::Of<T>();
