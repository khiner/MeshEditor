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
#include <format>
#include <map>
#include <unordered_map>
std::expected<std::vector<uint32_t>, std::string> ImportObjPlyMaterials(state::Scene &r, state::Entity viewport, std::span<const ObjPlyMaterial> materials, const std::filesystem::path &mesh_path) {
    struct LoadedTexture {
        DecodedImage Image;
        gltf::MimeType MimeType;
    };
    std::map<std::filesystem::path, LoadedTexture> loaded_textures;
    const auto texture_path_of = [&](const std::filesystem::path &path) {
        return (path.is_relative() ? mesh_path.parent_path() / path : path).lexically_normal();
    };
    const auto preload_texture = [&](const std::optional<std::filesystem::path> &source_path, std::string_view material_name, std::string_view texture_label) -> std::expected<void, std::string> {
        if (!source_path) return {};
        const auto path = texture_path_of(*source_path);
        if (loaded_textures.contains(path)) return {};
        const auto read = File::ReadAsString(path);
        if (!read) return std::unexpected{std::format("Cannot read OBJ texture '{}' for material '{}' ({}): {}", path.string(), material_name, texture_label, read.error())};
        const auto encoded = std::as_bytes(std::span{*read});
        auto decoded = DecodeImageRgba8(encoded, path.filename().string());
        if (!decoded) return std::unexpected{std::format("Cannot decode OBJ texture '{}': {}", path.string(), decoded.error())};
        loaded_textures.emplace(path, LoadedTexture{std::move(*decoded), gltf::detail::SniffMimeType(encoded)});
        return {};
    };
    for (const auto &source : materials) {
        if (const auto loaded = preload_texture(source.BaseColorTexturePath, source.Name, "baseColor"); !loaded) return std::unexpected{loaded.error()};
        if (const auto loaded = preload_texture(source.NormalTexturePath, source.Name, "normal"); !loaded) return std::unexpected{loaded.error()};
    }

    const auto &ctx = r.Context.get<const mtl::Context>();
    auto &slots = r.Context.get<mtl::BindlessSet>();
    auto &buffers = r.Context.get<GpuBuffers>();
    auto &textures = r.Context.get<TextureStore>();
    auto &sources = r.get_or_emplace<gltf::SourceAssets>(viewport);
    auto &manifest = r.get_or_emplace<MaterializedTextures>(viewport);
    const auto sampler_index = uint32_t(sources.Samplers.size());
    sources.Samplers.emplace_back(gltf::Sampler{.MagFilter = gltf::Filter::Nearest, .MinFilter = gltf::Filter::Nearest, .WrapS = gltf::Wrap::Repeat, .WrapT = gltf::Wrap::Repeat});

    auto obj_batch = BeginTextureUploadBatch(ctx);
    std::unordered_map<std::string, uint32_t> texture_slot_cache;
    std::unordered_map<uint32_t, uint32_t> source_texture_indices;
    const auto resolve_texture_slot =
        [&](
            const std::optional<std::filesystem::path> &source_texture_path,
            TextureColorSpace color_space
        ) -> uint32_t {
        if (!source_texture_path) return InvalidSlot;
        const auto texture_path = texture_path_of(*source_texture_path);

        const auto cache_key = std::format("{}|{}", texture_path.generic_string(), color_space == TextureColorSpace::Srgb ? "sRGB" : "Linear");
        if (const auto it = texture_slot_cache.find(cache_key); it != texture_slot_cache.end()) return it->second;

        const auto &loaded = loaded_textures.at(texture_path);
        const auto &decoded = loaded.Image;
        const auto sampler_slot = AllocateSamplerSlot(slots);
        auto texture = CreateTextureEntry(
            ctx, obj_batch, slots, sampler_slot, Rgba8Pixels{decoded.Pixels, decoded.Width, decoded.Height},
            TextureParams{
                .ColorSpace = color_space,
                .WrapS = MTL::SamplerAddressModeRepeat,
                .WrapT = MTL::SamplerAddressModeRepeat,
                .Sampler = SamplerConfig{},
                .Name = std::format("{} ({})", texture_path.filename().string(), color_space == TextureColorSpace::Srgb ? "sRGB" : "Linear"),
            },
            r.Context.get<const ActiveSamplerAnisotropy>().Value
        );
        const auto image_index = uint32_t(sources.Images.size());
        sources.Images.emplace_back(gltf::Image{
            .MimeType = loaded.MimeType,
            .Source = gltf::Image::SourceKind::External,
            .Name = texture_path.filename().string(),
            .SourcePath = project::AssetReference(r, texture_path).string(),
        });
        texture.SourceImageIndex = image_index;
        source_texture_indices.emplace(sampler_slot, uint32_t(sources.Textures.size()));
        sources.Textures.emplace_back(gltf::Texture{.SamplerIndex = sampler_index, .ImageIndex = image_index, .Name = texture.Params.Name});
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
        const auto base_color_texture = resolve_texture_slot(source.BaseColorTexturePath, TextureColorSpace::Srgb);
        const auto normal_texture = resolve_texture_slot(source.NormalTexturePath, TextureColorSpace::Linear);
        scene_material_indices[material_index] = buffers.Materials.Append(PBRMaterial{
            .BaseColorFactor = source.BaseColorFactor,
            .MetallicFactor = std::clamp(source.MetallicFactor, 0.f, 1.f),
            .RoughnessFactor = std::clamp(source.RoughnessFactor, 0.f, 1.f),
            .AlphaMode = (source.BaseColorFactor.w < 1.f || source.HasAlphaTexture) ? MaterialAlphaMode::Blend : MaterialAlphaMode::Opaque,
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

    auto &material_store = r.Context.get<MaterialStore>();
    material_store.AppendNames(std::move(names));

    return scene_material_indices;
}
