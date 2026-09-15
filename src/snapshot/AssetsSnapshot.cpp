#include "gltf/GltfScene.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "snapshot/SnapshotRegistration.h"

namespace snapshot::detail {

void RegisterAssets(Tables &tables) {
    Persistent<MaterialVariants, MaterializedTextures, PbrMeshFeatures, GltfNode, SourceIndex, MeshSourceLayout, gltf::SourceAssets>(tables);
}
} // namespace snapshot::detail
