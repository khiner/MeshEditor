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
#include "timeline/TimelineTheme.h"

namespace snapshot::detail {

void RegisterScene(Tables &tables) {
    Classify(
        tables,
        TypeList<
            Transform, PunctualLight, Name, Selected, Active, ObjectKind, Scene, SceneMembership, SubElementOf,
            ScaleLocked, Instance, Hidden, SceneParent, SceneChildren, ObjectExtrasTag, Path, Perspective, Orthographic, TimelineRange,
            TimelinePlayback, TimelineNavigation, TimelineView, TimelineTheme, AnimationClips, Animations, MorphWeightRange, ImageLight, Visibility>{},
        TypeList<
            RenderInstance, WorldTransform, ModelsBuffer, LastEvaluatedFrame, PlaybackFrame,
            PosedLayout, MeshEditWork, MeshVertexOverlays, BoundsRun,
            MeshletPositionsDirty, ClusterLodDirty, ClusterLodDemand, MeshDisplayDirty, MeshLayoutDirty, MeshBoundsDirty>{}
    );
}
} // namespace snapshot::detail
