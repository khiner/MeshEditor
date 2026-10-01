#include "CameraTypes.h"
#include "action/Core.h"
#include "action/Mesh.h"
#include "gizmo/GizmoInteraction.h"
#include "gpu/ViewportTheme.h"
#include "gpu/WorkspaceLights.h"
#include "selection/SelectionComponents.h"
#include "snapshot/SnapshotRegistration.h"
#include "viewport/GizmoDrag.h"
#include "viewport/InteractionComponents.h"
#include "viewport/VideoRecording.h"
#include "viewport/ViewCamera.h"
#include "viewport/ViewCameraSerialize.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"

namespace snapshot::detail {
// ViewCamera requires a constructor seed before deserialization.
void EmplaceViewCamera(state::Scene &r, state::Entity e, std::span<const std::byte> bytes) {
    ViewCamera v{vec3{0, 0, 1}, vec3{0}, CameraLens{}};
    if (zpp::bits::failure(zpp::bits::in{bytes}(v))) return;
    r.emplace_or_replace<ViewCamera>(e, v);
}
template<> inline constexpr auto CustomEmplace<ViewCamera> = &EmplaceViewCamera;

void RegisterViewport(Tables &tables) {
    Persistent<
        ViewportTheme, WorkspaceLights, ViewCamera, LookingThrough, Interaction, EditMode, OrbitToActive,
        ViewportDisplay, MaterialPreviewLighting, RenderedLighting, StudioEnvironment, TransformGizmoState>(tables);
    tables.Snapshots[state::Type<ViewCamera>()].History = false;
    Derived<
        SavedViewCamera, EnabledInteractionModes, AdditiveBoxSelectBaseline, ExciteSelectionBaseline,
        PendingEditElementClick, PendingBoxSelect, PendingPick, BoxSelectState,
        GizmoInteraction, PendingTransform, StartPivot, StartScreenTransform, StartTransform, StartBoneLength, action::DragFieldStart, VideoRecording>(tables);
}
} // namespace snapshot::detail
