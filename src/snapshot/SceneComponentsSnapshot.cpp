#include "CameraTypes.h"
#include "Path.h"
#include "animation/AnimationData.h"
#include "animation/MorphWeights.h"
#include "mesh/MeshComponents.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/LightComponents.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "snapshot/SnapshotRegistration.h"
#include "timeline/Timeline.h"

namespace snapshot::detail {

void RegisterScene(Tables &tables) {
    Persistent<
        Transform, PunctualLight, Name, Selected, Active, ObjectKind, Scene, ActiveScene, SceneMembership, SubElementOf,
        ScaleLocked, Instance, Hidden, SceneParent, SceneChildren, ObjectExtrasTag, Path, Perspective, Orthographic, TimelineRange,
        TimelinePlayback, TimelineNavigation, TimelineView, AnimationClips, Animations, MorphWeightRange, ImageLight, Visibility>(tables);
    Derived<
        RenderInstance, WorldTransform, ModelsBuffer, LightIndex, LastEvaluatedFrame,
        PlaybackFrame>(tables);
    Derived<
        PosedLayout, MeshEditWork, MeshVertexOverlays, BoundsRun,
        MeshletPositionsDirty, ClusterLodDirty, ClusterLodDemand, MeshDisplayDirty, MeshLayoutDirty, MeshBoundsDirty>(tables);
}
} // namespace snapshot::detail
