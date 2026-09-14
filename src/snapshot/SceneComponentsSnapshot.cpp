#include "CameraTypes.h"
#include "Path.h"
#include "animation/AnimationData.h"
#include "animation/AnimationTimeline.h"
#include "animation/MorphWeightState.h"
#include "armature/ArmatureComponents.h"
#include "gpu/PunctualLight.h"
#include "gpu/Transform.h"
#include "mesh/MeshComponents.h"
#include "render/Instance.h"
#include "render/LightComponents.h"
#include "render/MaterialComponents.h"
#include "render/MeshBuffers.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "scene/WorldTransform.h"
#include "snapshot/SnapshotRegistration.h"
#include "viewport/ViewportEvents.h"

namespace snapshot::detail {
template<> inline constexpr auto SkipEntityFor<Transform> = [](const state::Scene &r, state::Entity e) { return r.all_of<BoneIndex>(e); };
template<> inline constexpr bool ForceFieldwise<Camera> = true;
template<> inline constexpr bool ForceFieldwise<TimelinePlayback> = true;

void RegisterScene(Tables &tables) {
    Persistent<
        Transform, PunctualLight, Name, Selected, Active, ObjectKind, Scene, ActiveScene, SceneMembership, SubElementOf,
        ScaleLocked, Instance, Hidden, SceneNode, ParentInverse, ObjectExtrasTag, Path, Camera, TimelineRange,
        TimelinePlayback, AnimationTimelineView, NodeTransformAnimation, MorphWeightAnimation, MorphWeightState>(tables);
    // Transform is Persistent only without BoneIndex; capture it before that role changes.
    tables.Snapshots[state::Type<BoneIndex>()].CaptureWith = state::Type<Transform>();
    Derived<
        RenderInstance, WorldTransform, ModelsBuffer, MaterialDirty, LightIndex, LastEvaluatedFrame, MorphWeightGpuRange,
        PlaybackFrame>(tables);
}
} // namespace snapshot::detail
