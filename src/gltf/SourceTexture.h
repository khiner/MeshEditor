#pragma once

#include "image/ImageDecode.h"
#include "state/Entity.h"

#include <optional>

namespace gltf {
struct NormalMapRef {
    uint32_t Image{0}; // Index into SourceAssets::Images.
    uint32_t TexCoord{0};
    float Scale{1};
};

// Returns the resolved source image index.
std::optional<uint32_t> TextureImageIndex(const state::Scene &, uint32_t texture_index);

// Returns decoded RGBA8 source pixels.
std::optional<DecodedImage> DecodeImageRgba8(const state::Scene &, uint32_t image_index);

// Returns the normal map for the mesh's first primitive.
std::optional<NormalMapRef> MeshMaterialNormalMap(const state::Scene &, state::Entity mesh_entity);
} // namespace gltf
