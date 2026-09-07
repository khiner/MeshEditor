#include "render/MaterialImport.h"
#include "File.h"
#include "assets/MeshImport.h"
#include "render/GpuBuffers.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include <entt/entity/registry.hpp>
#include <iostream>
#include <unordered_map>
void ImportObjPlyMaterials(entt::registry &r, std::span<const ObjPlyMaterial> materials, const std::filesystem::path &mesh_path, uint32_t mesh_store_id) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    auto &slots = r.ctx().get<mtl::BindlessSet>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &meshes = r.ctx().get<MeshStore>();
    auto &textures = r.ctx().get<TextureStore>();

    auto obj_batch = BeginTextureUploadBatch(ctx, r.ctx().get<mtl::LibraryCache>());
    std::unordered_map<std::string, uint32_t> texture_slot_cache;
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

        auto texture = CreateTextureEntryFromEncoded(
            ctx,
            obj_batch,
            slots,
            std::as_bytes(std::span{encoded}),
            texture_path.filename().string(),
            std::format("{} ({})", texture_path.filename().string(), color_space == TextureColorSpace::Srgb ? "sRGB" : "Linear"),
            color_space,
            MTL::SamplerAddressModeRepeat,
            MTL::SamplerAddressModeRepeat,
            SamplerConfig{}, r.ctx().get<const ActiveSamplerAnisotropy>().Value
        );
        if (!texture) {
            std::cerr << std::format(
                "Warning: Failed to decode OBJ texture '{}' for material '{}' ({}) in '{}': {}\n",
                texture_path.string(), material_name, texture_label, mesh_path.string(), texture.error()
            );
            return InvalidSlot;
        }

        const auto sampler_slot = texture->SamplerSlot;
        textures.Textures.emplace_back(std::move(*texture));
        texture_slot_cache.emplace(cache_key, sampler_slot);
        return sampler_slot;
    };

    std::vector<uint32_t> scene_material_indices(materials.size(), 0u);
    std::vector<std::string> names;
    names.reserve(materials.size());
    buffers.Materials.ReserveElements(buffers.Materials.Count() + materials.size());
    for (uint32_t material_index = 0; material_index < materials.size(); ++material_index) {
        const auto &source = materials[material_index];
        const auto material_name = source.Name.empty() ? std::format("Material{}", material_index) : source.Name;
        const auto base_color_texture = resolve_texture_slot(source.BaseColorTexturePath, TextureColorSpace::Srgb, material_name, "baseColor");
        const auto normal_texture = resolve_texture_slot(source.NormalTexturePath, TextureColorSpace::Linear, material_name, "normal");
        scene_material_indices[material_index] = buffers.Materials.Append({
            .BaseColorFactor = source.BaseColorFactor,
            .MetallicFactor = std::clamp(source.MetallicFactor, 0.f, 1.f),
            .RoughnessFactor = std::clamp(source.RoughnessFactor, 0.f, 1.f),
            .AlphaMode = (source.BaseColorFactor.w < 1.f || source.HasAlphaTexture) ?
                MaterialAlphaMode::Blend :
                MaterialAlphaMode::Opaque,
            .BaseColorTexture = {.Slot = base_color_texture != InvalidSlot ? base_color_texture : textures.WhiteTextureSlot},
            .NormalTexture = {.Slot = normal_texture},
        });
        names.emplace_back(material_name);
    }
    SubmitTextureUploadBatch(obj_batch);

    auto &material_store = r.ctx().get<MaterialStore>();
    material_store.Names.insert(material_store.Names.end(), std::make_move_iterator(names.begin()), std::make_move_iterator(names.end()));

    if (auto primitive_materials = meshes.GetPrimitiveMaterialIndices(mesh_store_id); !primitive_materials.empty()) {
        const auto fallback = scene_material_indices.front();
        for (auto &primitive_material : primitive_materials) {
            primitive_material = primitive_material < scene_material_indices.size() ? scene_material_indices[primitive_material] : fallback;
        }
    }
}
