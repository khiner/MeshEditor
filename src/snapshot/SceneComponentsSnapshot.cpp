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

void RegisterScene(Tables &tables) {
    Persistent<
        Transform, PunctualLight, Name, Selected, Active, ObjectKind, Scene, ActiveScene, SceneMembership, SubElementOf,
        ScaleLocked, Instance, Hidden, SceneNode, ObjectExtrasTag, Path, Camera, TimelineRange,
        TimelinePlayback, TimelineNavigation, AnimationTimelineView, NodeTransformAnimation, MorphWeightAnimation, MorphWeightState>(tables);
    Derived<
        RenderInstance, WorldTransform, ModelsBuffer, LightIndex, LastEvaluatedFrame, MorphWeightGpuRange,
        PlaybackFrame>(tables);
}
} // namespace snapshot::detail
