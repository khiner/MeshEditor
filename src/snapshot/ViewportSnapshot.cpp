#include "action/Core.h"
#include "gizmo/GizmoInteraction.h"
#include "gpu/ViewportTheme.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionState.h"
#include "snapshot/SnapshotRegistration.h"
#include "ui/InterfaceTheme.h"
#include "viewport/GizmoDrag.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"
#include "viewport/ViewportScene.h"

namespace snapshot::detail {
void RegisterViewport(Tables &tables) {
    Classify(
        tables,
        TypeList<ViewportTheme, InterfaceTheme, Interaction, EditMode, AnisotropicFilter, TransformGizmoState, ViewportScene>{},
        TypeList<
            EnabledInteractionModes, AdditiveBoxSelectBaseline, ExciteSelectionBaseline,
            PendingEditElementClick, PendingBoxSelect, PendingPick, PendingViewSetup, BoxSelectState,
            GizmoInteraction, PendingTransform, StartPivot, StartScreenTransform, StartTransform, StartBoneLength, action::DragFieldStart,
            TransformRoots, EditPrimaries, SelectionFlags, SelectedKeyframes, OutlinerRows>{}
    );
}
} // namespace snapshot::detail
