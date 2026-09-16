#include "CameraTypes.h"
#include "action/Core.h"
#include "gizmo/GizmoInteraction.h"
#include "gpu/ViewportTheme.h"
#include "gpu/WorkspaceLights.h"
#include "scene/RotationUi.h"
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
// ViewCamera and LookingThrough require constructor seeds before deserialization.
void EmplaceViewCamera(state::Scene &r, state::Entity e, std::span<const std::byte> bytes) {
    ViewCamera v{vec3{0, 0, 1}, vec3{0}, Camera{}};
    if (zpp::bits::failure(zpp::bits::in{bytes}(v))) return;
    r.emplace_or_replace<ViewCamera>(e, v);
}
void EmplaceLookingThrough(state::Scene &r, state::Entity e, std::span<const std::byte> bytes) {
    LookingThrough l{ViewCamera{vec3{0, 0, 1}, vec3{0}, Camera{}}};
    if (zpp::bits::failure(zpp::bits::in{bytes}(l))) return;
    r.emplace_or_replace<LookingThrough>(e, std::move(l));
}

template<> inline constexpr auto CustomEmplace<ViewCamera> = &EmplaceViewCamera;
template<> inline constexpr auto CustomEmplace<LookingThrough> = &EmplaceLookingThrough;

void RegisterViewport(Tables &tables) {
    Persistent<
        ViewportTheme, WorkspaceLights, ViewCamera, LookingThrough, Interaction, EditMode, OrbitToActive,
        ViewportDisplay, MaterialPreviewLighting, RenderedLighting, StudioEnvironment, TransformGizmoState>(tables);
    tables.Snapshots[state::Type<ViewCamera>()].History = false;
    Derived<
        EnabledInteractionModes, AdditiveBoxSelectBaseline, ExciteSelectionBaseline,
        PendingEditElementClick, PendingBoxSelect, PendingPick, BoxSelectState, RotationUiVariant, RotationUiDriving,
        GizmoInteraction, PendingTransform, StartScreenTransform, StartTransform, StartBoneLength, action::DragFieldStart, VideoRecording>(tables);
}
} // namespace snapshot::detail
