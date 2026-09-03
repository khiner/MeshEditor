#include "SourceTexture.h"

#include "File.h"
#include "SourceAssets.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"
#include "render/GpuBuffers.h"
#include "render/Textures.h"

#include <entt/entity/registry.hpp>

namespace gltf {
namespace {
const SourceAssets *Assets(const entt::registry &r) {
    const auto view = r.view<const SourceAssets>();
    return view.empty() ? nullptr : &view.get<const SourceAssets>(view.front());
}
} // namespace

std::optional<uint32_t> TextureImageIndex(const entt::registry &r, uint32_t texture_index) {
    const auto *assets = Assets(r);
    if (!assets || texture_index >= assets->Textures.size()) return {};
    const auto image_index = ResolveImageIndex(assets->Textures[texture_index]);
    if (!image_index || *image_index >= assets->Images.size()) return {};
    return image_index;
}

std::optional<DecodedImage> DecodeImageRgba8(const entt::registry &r, uint32_t image_index) {
    const auto *assets = Assets(r);
    if (!assets || image_index >= assets->Images.size()) return {};
    const auto &image = assets->Images[image_index];
    const auto decode = [&](std::span<const std::byte> bytes) -> std::optional<DecodedImage> {
        auto decoded = ::DecodeImageRgba8(bytes, image.Name);
        return decoded ? std::optional{std::move(*decoded)} : std::nullopt;
    };
    if (!image.Bytes.empty()) return decode(image.Bytes);
    // Reload external sources from their retained path.
    const auto bytes = File::Read(image.SourceAbsPath);
    return bytes ? decode(*bytes) : std::nullopt;
}

std::optional<NormalMapRef> MeshMaterialNormalMap(const entt::registry &r, entt::entity mesh_entity) {
    const auto mesh = TryGetMesh(r, mesh_entity);
    if (!mesh) return {};
    const auto materials = r.ctx().get<const MeshStore>().GetPrimitiveMaterialIndices(mesh->GetStoreId());
    if (materials.empty()) return {};

    // Material bindings contain the remapped GPU indices.
    const auto &buffers = r.ctx().get<const GpuBuffers>();
    const uint32_t material = materials.front();
    if (material >= buffers.Materials.Count()) return {};
    const auto &pbr = buffers.Materials.Data()[material];
    if (pbr.NormalTexture.Slot == InvalidSlot) return {};

    // Resolve the uploaded source image through the sampler slot.
    for (auto [_, manifest] : r.view<const MaterializedTextures>().each()) {
        for (const auto &t : manifest.Items) {
            if (t.SamplerSlot != pbr.NormalTexture.Slot) continue;
            return NormalMapRef{.Image = t.SourceImageIndex, .TexCoord = pbr.NormalTexture.TexCoord, .Scale = pbr.NormalScale};
        }
    }
    return {};
}
} // namespace gltf
