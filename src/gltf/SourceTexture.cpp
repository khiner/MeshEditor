#include "SourceTexture.h"
#include "project/Assets.h"

#include "File.h"
#include "SourceAssets.h"
#include "mesh/MeshStore.h"
#include "render/GpuBuffers.h"
#include "render/Textures.h"

#include "state/Registry.h"

namespace gltf {
namespace {
const SourceAssets *Assets(const state::Registry &r) {
    const auto view = r.view<const SourceAssets>();
    return view.empty() ? nullptr : &r.get<const SourceAssets>(view.front());
}
} // namespace

std::optional<uint32_t> TextureImageIndex(const state::Registry &r, uint32_t texture_index) {
    const auto *assets = Assets(r);
    if (!assets || texture_index >= assets->Textures.size()) return {};
    const auto image_index = ResolveImageIndex(assets->Textures[texture_index]);
    if (!image_index || *image_index >= assets->Images.size()) return {};
    return image_index;
}

std::optional<DecodedImage> DecodeImageRgba8(const state::Registry &r, uint32_t image_index) {
    const auto *assets = Assets(r);
    if (!assets || image_index >= assets->Images.size()) return {};
    const auto &image = assets->Images[image_index];
    const auto decode = [&](std::span<const std::byte> bytes) -> std::optional<DecodedImage> {
        auto decoded = ::DecodeImageRgba8(bytes, image.Name);
        return decoded ? std::optional{std::move(*decoded)} : std::nullopt;
    };
    if (!image.Bytes.empty()) return decode(image.Bytes);
    // Reload external sources from their retained path.
    const auto bytes = File::Read(project::ResolveAsset(r, image.SourcePath));
    return bytes ? decode(*bytes) : std::nullopt;
}

std::optional<NormalMapRef> MeshMaterialNormalMap(const state::Registry &r, state::Entity mesh_entity) {
    const auto mesh = TryGetMesh(r, mesh_entity);
    if (!mesh) return {};
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto materials = meshes.Arenas().PrimitiveMaterials.Get(meshes.Get(mesh->GetStoreId()).PrimitiveMaterials);
    if (materials.empty()) return {};

    // Material bindings contain the remapped GPU indices.
    const auto &buffers = r.Resources.get<const GpuBuffers>();
    const uint32_t material = materials.front();
    if (material >= buffers.Materials.Count<PBRMaterial>()) return {};
    const auto &pbr = buffers.Materials.GetSpan<PBRMaterial>()[material];
    if (pbr.NormalTexture.Slot == InvalidSlot) return {};

    // Resolve the uploaded source image through the sampler slot.
    const auto texture = TextureAt(r, pbr.NormalTexture.Slot);
    if (texture == state::Null) return {};
    return NormalMapRef{.Image = r.get<const MaterializedTexture>(texture).SourceImageIndex, .TexCoord = pbr.NormalTexture.TexCoord, .Scale = pbr.NormalScale};
}
} // namespace gltf
