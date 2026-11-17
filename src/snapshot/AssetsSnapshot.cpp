#include "gltf/GltfScene.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "snapshot/SnapshotRegistration.h"

namespace snapshot::detail {

void RegisterAssets(Tables &tables) {
    Classify(tables, TypeList<Material, MaterialVariants, MaterializedTexture, PbrMeshFeatures, GltfNode, SourceIndex, MeshSourceLayout, HasVariantMappings, gltf::SourceAssets, gltf::MaterialSourceMeta>{}, TypeList<>{}, TypeList<>{});
}
} // namespace snapshot::detail
