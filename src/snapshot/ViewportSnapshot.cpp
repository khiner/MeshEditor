#include "action/Core.h"
#include "gizmo/GizmoInteraction.h"
#include "gpu/ViewportTheme.h"
#include "gpu/WorkspaceLights.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionState.h"
#include "snapshot/SnapshotRegistration.h"
#include "ui/InterfaceTheme.h"
#include "viewport/FrameState.h"
#include "viewport/GizmoDrag.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewCameraSerialize.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportComponents.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"
#include "viewport/ViewportScene.h"

namespace snapshot::detail {
template<> ViewCameraState CopyNative(const ViewCameraState &source) {
    auto value = source;
    value.Active.StopMoving();
    if (value.LookThroughSaved) value.LookThroughSaved->StopMoving();
    return value;
}
void RegisterViewport(Tables &tables) {
    Classify(
        tables,
        TypeList<ViewportTheme, InterfaceTheme, Interaction, EditMode, AnisotropicFilter, TransformGizmoState, ViewportScene>{},
        Concat(TypeList<ActiveViewport, ActionInputs>{}, ViewValues{}),
        TypeList<
            Viewport, EnabledInteractionModes, AdditiveBoxSelectBaseline, ExciteSelectionBaseline,
            PendingBoxSelect, PendingViewSetup, BoxSelectState,
            GizmoInteraction, PendingTransform, StartPivot, StartScreenTransform, StartTransform, StartBoneLength, action::DragFieldStart,
            TransformRoots, EditPrimaries, SelectionFlags, SelectedKeyframes, OutlinerRows>{}
    );
}
} // namespace snapshot::detail
