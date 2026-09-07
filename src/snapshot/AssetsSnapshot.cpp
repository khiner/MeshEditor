#include "gltf/GltfScene.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "snapshot/SnapshotRegistration.h"

namespace snapshot::detail {

void RegisterAssets(Tables &tables) {
    Persistent<
        MaterialVariants, MaterializedTextures, PbrMeshFeatures, SourceNodeIndex, SourceParentNodeIndex,
        SourceSiblingIndex, SourceMeshIndex, SourceCameraIndex, SourceLightIndex, SourcePhysicsMaterialIndex,
        SourceCollisionFilterIndex, SourcePhysicsJointDefIndex, SourceSceneIndex, SourceMeshKind, GltfObject, CameraName,
        LightName, SourceObjectName, MeshName, SourceMatrixTransform, SourceEmptyName, MeshSourceLayout,
        gltf::SourceAssets>(tables);
    Derived<PendingTextureUploads>(tables);
}
} // namespace snapshot::detail
