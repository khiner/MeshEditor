#pragma once

#include "entt_fwd.h"
#include "image/ImageDecode.h"

#include <optional>

namespace gltf {
// A normal map to sample, resolved to the source image holding its pixels.
struct NormalMapRef {
    uint32_t Image{0}; // Index into SourceAssets::Images.
    uint32_t TexCoord{0};
    float Scale{1};
};

// The source image a texture resolves to, or empty when the texture or its image is missing.
std::optional<uint32_t> TextureImageIndex(const entt::registry &, uint32_t texture_index);

// The decoded pixels of a source image, in RGBA8, or empty when it cannot be read.
// The scene's source assets sit on one entity, so the image resolves from the registry alone.
std::optional<DecodedImage> DecodeImageRgba8(const entt::registry &, uint32_t image_index);

// The normal map of the material on a mesh's first primitive, or empty when it has none.
// Materials and texture coordinate sets are both per primitive, so a mesh whose primitives carry different materials answers for one of them.
std::optional<NormalMapRef> MeshMaterialNormalMap(const entt::registry &, entt::entity mesh_entity);
} // namespace gltf
