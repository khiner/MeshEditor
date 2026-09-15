#include "assets/MaterialImport.h"
#include "File.h"
#include "assets/MeshImport.h"
#include "gltf/GltfConvert.h"
#include "image/ImageDecode.h"
#include "project/Assets.h"
#include "render/GpuBuffers.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "state/Scene.h"
#include <iostream>
#include <unordered_map>
void ImportObjPlyMaterials(state::Scene &r, state::Entity viewport, std::span<const ObjPlyMaterial> materials, const std::filesystem::path &mesh_path, uint32_t mesh_store_id) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    auto &slots = r.ctx().get<mtl::BindlessSet>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &meshes = r.ctx().get<MeshStore>();
    auto &textures = r.ctx().get<TextureStore>();
    auto &sources = r.get_or_emplace<gltf::SourceAssets>(viewport);
    auto &manifest = r.get_or_emplace<MaterializedTextures>(viewport);
    const auto sampler_index = uint32_t(sources.Samplers.size());
    sources.Samplers.emplace_back(gltf::Sampler{.MagFilter = gltf::Filter::Nearest, .MinFilter = gltf::Filter::Nearest, .WrapS = gltf::Wrap::Repeat, .WrapT = gltf::Wrap::Repeat, .Name = {}});

    auto obj_batch = BeginTextureUploadBatch(ctx);
    std::unordered_map<std::string, uint32_t> texture_slot_cache;
    std::unordered_map<uint32_t, uint32_t> source_texture_indices;
    const auto resolve_texture_slot =
        [&](
            const std::optional<std::filesystem::path> &source_texture_path,
            TextureColorSpace color_space,
            std::string_view material_name, std::string_view texture_label
        ) -> uint32_t {
        if (!source_texture_path) return InvalidSlot;
        auto texture_path = *source_texture_path;
        if (texture_path.is_relative()) texture_path = mesh_path.parent_path() / texture_path;
        texture_path = texture_path.lexically_normal();

        const auto cache_key = std::format("{}|{}", texture_path.generic_string(), color_space == TextureColorSpace::Srgb ? "sRGB" : "Linear");
        if (const auto it = texture_slot_cache.find(cache_key); it != texture_slot_cache.end()) return it->second;

        const auto read = File::ReadAsString(texture_path);
        if (!read) {
            std::cerr << std::format(
                "Warning: Failed to read OBJ texture '{}' for material '{}' ({}) in '{}': {}\n",
                texture_path.string(), material_name, texture_label, mesh_path.string(), read.error()
            );
            return InvalidSlot;
        }
        const std::string &encoded = *read;
        const auto decoded = DecodeImageRgba8(std::as_bytes(std::span{encoded}), texture_path.filename().string());
        if (!decoded) {
            std::cerr << std::format(
                "Warning: Failed to decode OBJ texture '{}' for material '{}' ({}) in '{}': {}\n",
                texture_path.string(), material_name, texture_label, mesh_path.string(), decoded.error()
            );
            return InvalidSlot;
        }
        const auto sampler_slot = AllocateSamplerSlot(slots);
        auto texture = CreateTextureEntry(
            ctx, obj_batch, slots, sampler_slot, Rgba8Pixels{decoded->Pixels, decoded->Width, decoded->Height},
            TextureParams{
                .ColorSpace = color_space,
                .WrapS = MTL::SamplerAddressModeRepeat,
                .WrapT = MTL::SamplerAddressModeRepeat,
                .Sampler = SamplerConfig{},
                .Name = std::format("{} ({})", texture_path.filename().string(), color_space == TextureColorSpace::Srgb ? "sRGB" : "Linear"),
            },
            r.ctx().get<const ActiveSamplerAnisotropy>().Value
        );
        const auto image_index = uint32_t(sources.Images.size());
        sources.Images.emplace_back(gltf::Image{
            .Bytes = {},
            .MimeType = gltf::detail::SniffMimeType(std::as_bytes(std::span{encoded})),
            .Source = gltf::Image::SourceKind::External,
            .SourceHadMimeType = false,
            .IsDirty = false,
            .Name = texture_path.filename().string(),
            .Uri = {},
            .SourcePath = project::AssetReference(r, texture_path).string(),
        });
        texture.SourceImageIndex = image_index;
        source_texture_indices.emplace(sampler_slot, uint32_t(sources.Textures.size()));
        sources.Textures.emplace_back(gltf::Texture{.SamplerIndex = sampler_index, .ImageIndex = image_index, .WebpImageIndex = {}, .BasisuImageIndex = {}, .DdsImageIndex = {}, .Name = texture.Params.Name});
        manifest.Items.emplace_back(MaterializedTexture{.SamplerSlot = sampler_slot, .SourceImageIndex = image_index, .Params = texture.Params});
        textures.Textures.emplace_back(std::move(texture));
        texture_slot_cache.emplace(cache_key, sampler_slot);
        return sampler_slot;
    };

    std::vector<uint32_t> scene_material_indices(materials.size(), 0u);
    std::vector<std::string> names;
    names.reserve(materials.size());
    buffers.Materials.Reserve((buffers.Materials.Count<PBRMaterial>() + materials.size()) * sizeof(PBRMaterial));
    for (uint32_t material_index = 0; material_index < materials.size(); ++material_index) {
        const auto &source = materials[material_index];
        const auto material_name = source.Name.empty() ? std::format("Material{}", material_index) : source.Name;
        const auto base_color_texture = resolve_texture_slot(source.BaseColorTexturePath, TextureColorSpace::Srgb, material_name, "baseColor");
        const auto normal_texture = resolve_texture_slot(source.NormalTexturePath, TextureColorSpace::Linear, material_name, "normal");
        scene_material_indices[material_index] = buffers.Materials.Append(PBRMaterial{
            .BaseColorFactor = source.BaseColorFactor,
            .MetallicFactor = std::clamp(source.MetallicFactor, 0.f, 1.f),
            .RoughnessFactor = std::clamp(source.RoughnessFactor, 0.f, 1.f),
            .AlphaMode = (source.BaseColorFactor.w < 1.f || source.HasAlphaTexture) ?
                MaterialAlphaMode::Blend :
                MaterialAlphaMode::Opaque,
            .BaseColorTexture = {.Slot = base_color_texture != InvalidSlot ? base_color_texture : textures.WhiteTextureSlot},
            .NormalTexture = {.Slot = normal_texture},
        });
        sources.MaterialMetas.resize(buffers.Materials.Count<PBRMaterial>() - 1);
        auto &meta = sources.MaterialMetas.back();
        meta = {};
        if (base_color_texture != InvalidSlot) meta.TextureSlots[MTS_BaseColor] = source_texture_indices.at(base_color_texture);
        if (normal_texture != InvalidSlot) meta.TextureSlots[MTS_Normal] = source_texture_indices.at(normal_texture);
        names.emplace_back(material_name);
    }
    SubmitTextureUploadBatch(obj_batch);

    auto &material_store = r.ctx().get<MaterialStore>();
    material_store.AppendNames(std::move(names));

    if (auto primitive_materials = meshes.EditPrimitiveMaterials(mesh_store_id); !primitive_materials.empty()) {
        const auto fallback = scene_material_indices.front();
        for (auto &primitive_material : primitive_materials) {
            primitive_material = primitive_material < scene_material_indices.size() ? scene_material_indices[primitive_material] : fallback;
        }
    }
}
