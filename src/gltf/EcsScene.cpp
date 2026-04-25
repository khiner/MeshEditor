#include "EcsScene.h"

#include "AnimationTimeline.h"
#include "Armature.h"
#include "Instance.h"
#include "NodeTransformAnimation.h"
#include "Path.h"
#include "PbrFeature.h"
#include "SceneOps.h"
#include "SceneTextures.h"
#include "SceneTree.h"
#include "Timer.h"
#include "TransformMath.h"
#include "mesh/MeshStore.h"
#include "mesh/MorphTargetData.h"
#include "scene_impl/SceneBuffers.h"
#include "scene_impl/SceneComponents.h"

#include <entt/entity/registry.hpp>
#include <iostream>
#include <unordered_set>

std::expected<gltf::PopulateResult, std::string> gltf::PopulateGltfScene(gltf::Scene &source, const std::filesystem::path &source_path, gltf::PopulateContext ctx) {
    const Timer timer{"PopulateGltfScene"};

    auto &R = ctx.R;
    const auto SceneEntity = ctx.SceneEntity;
    auto &texture_store = ctx.Textures;
    const auto texture_start = texture_store.Textures.size();
    const auto material_start = ctx.Buffers.Materials.Count();
    const auto material_name_start = R.get<const MaterialStore>(SceneEntity).Names.size();
    const auto rollback_import_side_effects = [&] {
        if (texture_store.Textures.size() > texture_start) {
            ReleaseSamplerSlots(ctx.Slots, CollectSamplerSlots(std::span<const TextureEntry>{texture_store.Textures}.subspan(texture_start)));
            texture_store.Textures.resize(texture_start);
        }
        if (ctx.Buffers.Materials.Count() > material_start) ctx.Buffers.Materials.SetCount(material_start);
        R.patch<MaterialStore>(
            SceneEntity,
            [&](auto &store) {
                if (store.Names.size() > material_name_start) store.Names.resize(material_name_start);
            }
        );
    };
    struct ImportRollbackGuard {
        decltype(rollback_import_side_effects) &Rollback;
        bool Enabled{true};
        ~ImportRollbackGuard() {
            if (Enabled) Rollback();
        }
    };
    ImportRollbackGuard import_rollback_guard{rollback_import_side_effects};

    const auto resolve_image_index = [&](const gltf::Texture &texture) -> std::optional<uint32_t> {
        if (texture.ImageIndex) return texture.ImageIndex;
        if (texture.WebpImageIndex) return texture.WebpImageIndex;
        if (texture.BasisuImageIndex) return texture.BasisuImageIndex;
        return texture.DdsImageIndex;
    };

    // Snapshot source-form gltf data onto SceneEntity before the materials loop remaps `tex.Slot`
    // from gltf texture index to bindless sampler slot.
    std::vector<std::string> animation_order;
    animation_order.reserve(source.Animations.size());
    for (const auto &a : source.Animations) animation_order.emplace_back(a.Name);
    R.emplace_or_replace<GltfSourceAssets>(
        SceneEntity,
        GltfSourceAssets{
            .Copyright = source.Copyright,
            .Generator = source.Generator,
            .MinVersion = source.MinVersion,
            .AssetExtras = source.AssetExtras,
            .AssetExtensions = source.AssetExtensions,
            .DefaultSceneName = source.DefaultSceneName,
            .DefaultSceneRoots = source.DefaultSceneRoots,
            .ExtensionsRequired = source.ExtensionsRequired,
            .MaterialVariants = source.MaterialVariants,
            .ExtrasByEntity = source.ExtrasByEntity,
            .Textures = source.Textures,
            .Images = source.Images,
            .Samplers = source.Samplers,
            .AnimationOrder = std::move(animation_order),
            .ImageBasedLight = source.ImageBasedLight,
        }
    );

    // Snapshot the FromGpu-lossy delta per source material before the texture-slot remap and
    // before extension-block optionality is folded into the GPU `PBRMaterial`.
    {
        const auto opt_slot = [](const auto &opt, auto field) -> uint32_t {
            return opt ? ((*opt).*field).Slot : InvalidSlot;
        };
        using M = MaterialSourceMeta;
        std::vector<M> metas;
        metas.reserve(source.Materials.size());
        for (const auto &nm : source.Materials) {
            const auto &m = nm.Value;
            M meta{
                .EmissiveStrength = m.EmissiveStrength,
                .BaseSlotMeta = {m.BaseColorMeta, m.MetallicRoughnessMeta, m.NormalMeta, m.OcclusionMeta, m.EmissiveMeta},
                .NameWasEmpty = nm.Name.empty(),
            };
            meta.TextureSlots = {
                m.BaseColorTexture.Slot,
                m.MetallicRoughnessTexture.Slot,
                m.NormalTexture.Slot,
                m.OcclusionTexture.Slot,
                m.EmissiveTexture.Slot,
                opt_slot(m.Specular, &::Specular::Texture),
                opt_slot(m.Specular, &::Specular::ColorTexture),
                opt_slot(m.Sheen, &::Sheen::ColorTexture),
                opt_slot(m.Sheen, &::Sheen::RoughnessTexture),
                opt_slot(m.Transmission, &::Transmission::Texture),
                opt_slot(m.DiffuseTransmission, &::DiffuseTransmission::Texture),
                opt_slot(m.DiffuseTransmission, &::DiffuseTransmission::ColorTexture),
                opt_slot(m.Volume, &::Volume::ThicknessTexture),
                opt_slot(m.Clearcoat, &::Clearcoat::Texture),
                opt_slot(m.Clearcoat, &::Clearcoat::RoughnessTexture),
                opt_slot(m.Clearcoat, &::Clearcoat::NormalTexture),
                opt_slot(m.Anisotropy, &::Anisotropy::Texture),
                opt_slot(m.Iridescence, &::Iridescence::Texture),
                opt_slot(m.Iridescence, &::Iridescence::ThicknessTexture),
            };
            meta.ExtensionPresence = uint16_t(
                (m.Ior ? M::ExtIor : 0) | (m.Dispersion ? M::ExtDispersion : 0) |
                (m.EmissiveStrength ? M::ExtEmissiveStrength : 0) |
                (m.Sheen ? M::ExtSheen : 0) | (m.Specular ? M::ExtSpecular : 0) |
                (m.Transmission ? M::ExtTransmission : 0) | (m.DiffuseTransmission ? M::ExtDiffuseTransmission : 0) |
                (m.Volume ? M::ExtVolume : 0) | (m.Clearcoat ? M::ExtClearcoat : 0) |
                (m.Anisotropy ? M::ExtAnisotropy : 0) | (m.Iridescence ? M::ExtIridescence : 0)
            );
            metas.emplace_back(std::move(meta));
        }
        R.patch<GltfSourceAssets>(SceneEntity, [&](auto &a) { a.MaterialMetas = std::move(metas); });
    }

    auto upload_batch = BeginTextureUploadBatch(ctx.Vk.Device, ctx.CommandPool, ctx.Buffers.Ctx);

    std::unordered_map<uint64_t, uint32_t> texture_slot_cache;
    // Cache on the resolved (image_index, sampler_index, color_space) rather than glTF texture index,
    // so that multiple glTF textures referencing the same image+sampler share a single TextureEntry and sampler slot.
    const auto texture_cache_key = [](uint32_t image_index, uint32_t sampler_index, TextureColorSpace color_space) {
        return (uint64_t(image_index) << 33u) | (uint64_t(sampler_index) << 1u) | (color_space == TextureColorSpace::Srgb ? 1u : 0u);
    };
    const auto resolve_texture_slot = [&](uint32_t texture_index, TextureColorSpace color_space) -> std::expected<uint32_t, std::string> {
        if (texture_index >= source.Textures.size()) return InvalidSlot;

        const auto &src_texture = source.Textures[texture_index];
        const auto image_index = resolve_image_index(src_texture);
        if (!image_index || *image_index >= source.Images.size()) return InvalidSlot;

        const auto sampler_index = src_texture.SamplerIndex.value_or(InvalidSlot);
        const auto cache_key = texture_cache_key(*image_index, sampler_index, color_space);
        if (const auto it = texture_slot_cache.find(cache_key); it != texture_slot_cache.end()) return it->second;

        const auto &src_image = source.Images[*image_index];
        const auto *src_sampler = src_texture.SamplerIndex && *src_texture.SamplerIndex < source.Samplers.size() ?
            &source.Samplers[*src_texture.SamplerIndex] :
            nullptr;
        static constexpr auto ToSamplerAddressMode = [](gltf::Wrap wrap) {
            switch (wrap) {
                case gltf::Wrap::ClampToEdge: return vk::SamplerAddressMode::eClampToEdge;
                case gltf::Wrap::MirroredRepeat: return vk::SamplerAddressMode::eMirroredRepeat;
                case gltf::Wrap::Repeat: return vk::SamplerAddressMode::eRepeat;
            }
            return vk::SamplerAddressMode::eRepeat;
        };
        static constexpr auto ToSamplerConfig = [](const gltf::Sampler *sampler) -> SamplerConfig {
            if (!sampler) return {.MinFilter = vk::Filter::eLinear, .MagFilter = vk::Filter::eLinear, .MipmapMode = vk::SamplerMipmapMode::eLinear, .UsesMipmaps = true};

            const auto mag_filter = sampler->MagFilter && *sampler->MagFilter == gltf::Filter::Nearest ? vk::Filter::eNearest : vk::Filter::eLinear;
            switch (sampler->MinFilter.value_or(gltf::Filter::LinearMipMapLinear)) {
                case gltf::Filter::Nearest:
                    return {.MinFilter = vk::Filter::eNearest, .MagFilter = mag_filter, .MipmapMode = vk::SamplerMipmapMode::eNearest, .UsesMipmaps = false};
                case gltf::Filter::Linear:
                    return {.MinFilter = vk::Filter::eLinear, .MagFilter = mag_filter, .MipmapMode = vk::SamplerMipmapMode::eNearest, .UsesMipmaps = false};
                case gltf::Filter::NearestMipMapNearest:
                    return {.MinFilter = vk::Filter::eNearest, .MagFilter = mag_filter, .MipmapMode = vk::SamplerMipmapMode::eNearest, .UsesMipmaps = true};
                case gltf::Filter::LinearMipMapNearest:
                    return {.MinFilter = vk::Filter::eLinear, .MagFilter = mag_filter, .MipmapMode = vk::SamplerMipmapMode::eNearest, .UsesMipmaps = true};
                case gltf::Filter::NearestMipMapLinear:
                    return {.MinFilter = vk::Filter::eNearest, .MagFilter = mag_filter, .MipmapMode = vk::SamplerMipmapMode::eLinear, .UsesMipmaps = true};
                case gltf::Filter::LinearMipMapLinear:
                    return {.MinFilter = vk::Filter::eLinear, .MagFilter = mag_filter, .MipmapMode = vk::SamplerMipmapMode::eLinear, .UsesMipmaps = true};
            }
            return {.MinFilter = vk::Filter::eLinear, .MagFilter = mag_filter, .MipmapMode = vk::SamplerMipmapMode::eLinear, .UsesMipmaps = true};
        };

        const auto sampler_config = ToSamplerConfig(src_sampler);
        const auto wrap_s = src_sampler ? ToSamplerAddressMode(src_sampler->WrapS) : vk::SamplerAddressMode::eRepeat;
        const auto wrap_t = src_sampler ? ToSamplerAddressMode(src_sampler->WrapT) : vk::SamplerAddressMode::eRepeat;
        const auto texture_name = std::format("{} ({})", src_texture.Name.empty() ? std::format("Texture{}", texture_index) : src_texture.Name, color_space == TextureColorSpace::Srgb ? "sRGB" : "Linear");

        auto texture = CreateTextureEntryFromImage(ctx.Vk, upload_batch, ctx.Slots, src_image, texture_name, color_space, wrap_s, wrap_t, sampler_config);
        if (!texture) return std::unexpected{std::move(texture.error())};
        const auto sampler_slot = texture->SamplerSlot;
        texture_store.Textures.emplace_back(std::move(*texture));
        texture_slot_cache.emplace(cache_key, sampler_slot);
        return sampler_slot;
    };

    std::vector<uint32_t> material_indices_by_gltf_material(source.Materials.size(), 0u);
    const auto material_count = ctx.Buffers.Materials.Count();
    const auto default_material_index = material_count > 0 ? material_count - 1u : 0u;
    std::vector<std::string> material_names;
    material_names.reserve(source.Materials.size());
    ctx.Buffers.Materials.Reserve(material_count + source.Materials.size());
    for (uint32_t material_index = 0; material_index < source.Materials.size(); ++material_index) {
        const auto &src_named_material = source.Materials[material_index];
        const auto &src_material = src_named_material.Value;
        const auto material_name = src_named_material.Name.empty() ? std::format("Material{}", material_index) : src_named_material.Name;
        const auto clamp_uv_set = [&](uint32_t uv_set, std::string_view texture_label) {
            if (uv_set <= 3u) return uv_set;
            std::cerr << std::format(
                "Warning: glTF material '{}' texture '{}' uses TEXCOORD_{}. MeshEditor currently supports TEXCOORD_0..3. Clamping to TEXCOORD_3.\n",
                material_name, texture_label, uv_set
            );
            return 3u;
        };
        // Resolves a texture slot in-place: tex.Slot starts as a glTF texture index (stored by GltfLoader),
        // and is replaced with the bindless sampler slot. UV fields are already correct from the loader.
        const auto resolve_texture = [&](TextureInfo &tex, TextureColorSpace color_space, std::string_view texture_label) -> std::expected<void, std::string> {
            if (tex.Slot == InvalidSlot) return {};
            const uint32_t gltf_index = tex.Slot;
            tex.TexCoord = clamp_uv_set(tex.TexCoord, texture_label);
            auto texture_slot_result = resolve_texture_slot(gltf_index, color_space);
            if (!texture_slot_result) return std::unexpected{std::move(texture_slot_result.error())};
            tex.Slot = *texture_slot_result;
            return {};
        };
        auto gpu_material = gltf::ToGpu(src_material);
        if (auto result = [&]() -> std::expected<void, std::string> {
                std::expected<void, std::string> resolve_result{};
                const auto check = [&](TextureInfo &tex, TextureColorSpace color_space, std::string_view texture_label) {
                    resolve_result = resolve_texture(tex, color_space, texture_label);
                    return resolve_result.has_value();
                };
                if (
                    !check(gpu_material.BaseColorTexture, TextureColorSpace::Srgb, "baseColor") ||
                    !check(gpu_material.MetallicRoughnessTexture, TextureColorSpace::Linear, "metallicRoughness") ||
                    !check(gpu_material.NormalTexture, TextureColorSpace::Linear, "normal") ||
                    !check(gpu_material.OcclusionTexture, TextureColorSpace::Linear, "occlusion") ||
                    !check(gpu_material.EmissiveTexture, TextureColorSpace::Srgb, "emissive") ||
                    !check(gpu_material.Specular.Texture, TextureColorSpace::Linear, "specular") ||
                    !check(gpu_material.Specular.ColorTexture, TextureColorSpace::Srgb, "specularColor") ||
                    !check(gpu_material.Sheen.ColorTexture, TextureColorSpace::Srgb, "sheenColor") ||
                    !check(gpu_material.Sheen.RoughnessTexture, TextureColorSpace::Linear, "sheenRoughness") ||
                    !check(gpu_material.Transmission.Texture, TextureColorSpace::Linear, "transmission") ||
                    !check(gpu_material.DiffuseTransmission.Texture, TextureColorSpace::Linear, "diffuseTransmission") ||
                    !check(gpu_material.DiffuseTransmission.ColorTexture, TextureColorSpace::Srgb, "diffuseTransmissionColor") ||
                    !check(gpu_material.Volume.ThicknessTexture, TextureColorSpace::Linear, "thickness") ||
                    !check(gpu_material.Clearcoat.Texture, TextureColorSpace::Linear, "clearcoat") ||
                    !check(gpu_material.Clearcoat.RoughnessTexture, TextureColorSpace::Linear, "clearcoatRoughness") ||
                    !check(gpu_material.Clearcoat.NormalTexture, TextureColorSpace::Linear, "clearcoatNormal") ||
                    !check(gpu_material.Anisotropy.Texture, TextureColorSpace::Linear, "anisotropy") ||
                    !check(gpu_material.Iridescence.Texture, TextureColorSpace::Linear, "iridescence") ||
                    !check(gpu_material.Iridescence.ThicknessTexture, TextureColorSpace::Linear, "iridescenceThickness")
                ) {
                    return std::unexpected{std::move(resolve_result.error())};
                }
                return {};
            }();
            !result) {
            return std::unexpected{std::move(result.error())};
        }
        material_indices_by_gltf_material[material_index] = ctx.Buffers.Materials.Append(gpu_material);
        material_names.emplace_back(material_name);
    }
    const auto fallback_material_index = material_indices_by_gltf_material.empty() ? default_material_index : material_indices_by_gltf_material.back();
    if (!material_names.empty()) {
        R.patch<MaterialStore>(
            SceneEntity,
            [&](auto &store) {
                store.Names.insert(store.Names.end(), std::make_move_iterator(material_names.begin()), std::make_move_iterator(material_names.end()));
            }
        );
    }

    SubmitTextureUploadBatch(upload_batch, ctx.Vk.Queue, ctx.OneShotFence, ctx.Vk.Device);

    // Pre-reserve MeshStore arenas to avoid O(N) reallocations during bulk mesh creation.
    {
        auto plan = [&](const std::optional<::MeshData> &data) { if (data) ctx.Meshes.PlanCreate(*data); };
        for (const auto &scene_mesh : source.Meshes) {
            if (scene_mesh.Triangles) {
                uint32_t morph_targets = scene_mesh.MorphData ? scene_mesh.MorphData->TargetCount : 0;
                ctx.Meshes.PlanCreate(*scene_mesh.Triangles, scene_mesh.TrianglePrimitives, scene_mesh.DeformData.has_value(), morph_targets);
            }
            plan(scene_mesh.Lines);
            plan(scene_mesh.Points);
        }
        ctx.Meshes.CommitReserves();
    }

    std::vector<entt::entity> mesh_entities;
    mesh_entities.reserve(source.Meshes.size());
    // Track morph data per mesh for later component setup
    std::vector<std::optional<MorphTargetData>> mesh_morphs;
    mesh_morphs.reserve(source.Meshes.size());
    entt::entity first_mesh_entity = entt::null;
    // Per-mesh: optional line entity + optional point entity
    struct ExtraPrimitiveEntities {
        entt::entity Lines{entt::null}, Points{entt::null};
    };
    std::vector<ExtraPrimitiveEntities> extra_entities_per_mesh(source.Meshes.size());
    for (uint32_t mi = 0; mi < source.Meshes.size(); ++mi) {
        auto &scene_mesh = source.Meshes[mi];
        entt::entity mesh_entity = entt::null;
        if (scene_mesh.Triangles) {
            // Detect PBR extension features before material index remapping.
            PbrFeatureMask mesh_pbr_mask{0};
            for (const auto gltf_mat_idx : scene_mesh.TrianglePrimitives.MaterialIndices) {
                if (gltf_mat_idx < source.Materials.size()) {
                    const auto mat = gltf::ToGpu(source.Materials[gltf_mat_idx].Value);
                    if (mat.Transmission.Factor > 0.f || mat.Transmission.Texture.Slot != InvalidSlot) mesh_pbr_mask |= PbrFeature::Transmission;
                    if (mat.DiffuseTransmission.Factor > 0.f || mat.DiffuseTransmission.Texture.Slot != InvalidSlot) mesh_pbr_mask |= PbrFeature::DiffuseTrans;
                    if (mat.Clearcoat.Factor > 0.f || mat.Clearcoat.Texture.Slot != InvalidSlot) mesh_pbr_mask |= PbrFeature::Clearcoat;
                    if (mat.Sheen.RoughnessFactor > 0.f || mat.Sheen.ColorTexture.Slot != InvalidSlot) mesh_pbr_mask |= PbrFeature::Sheen;
                    if (mat.Anisotropy.Strength != 0.f || mat.Anisotropy.Texture.Slot != InvalidSlot) mesh_pbr_mask |= PbrFeature::Anisotropy;
                    if (mat.Iridescence.Factor > 0.f || mat.Iridescence.Texture.Slot != InvalidSlot) mesh_pbr_mask |= PbrFeature::Iridescence;
                }
            }
            for (auto &local_material_index : scene_mesh.TrianglePrimitives.MaterialIndices) {
                local_material_index = local_material_index < material_indices_by_gltf_material.size() ?
                    material_indices_by_gltf_material[local_material_index] :
                    fallback_material_index;
            }
            // Snapshot per-primitive metadata + morph tangent deltas before move-into-CreateMesh
            // discards them. Stored as a sidecar so `BuildGltfScene` can re-emit verbatim.
            MeshSourceLayout layout{
                .VertexCounts = scene_mesh.TrianglePrimitives.VertexCounts,
                .AttributeFlags = scene_mesh.TrianglePrimitives.AttributeFlags,
                .HasSourceIndices = scene_mesh.TrianglePrimitives.HasSourceIndices,
                .VariantMappings = scene_mesh.TrianglePrimitives.VariantMappings,
                .Colors0ComponentCount = scene_mesh.TriangleAttrs.Colors0ComponentCount,
                .MorphTangentDeltas = scene_mesh.MorphData ? scene_mesh.MorphData->TangentDeltas : std::vector<vec3>{},
            };
            auto morph_data_copy = scene_mesh.MorphData; // Keep a copy for component setup
            auto mesh = ctx.Meshes.CreateMesh(
                std::move(*scene_mesh.Triangles), std::move(scene_mesh.TriangleAttrs), std::move(scene_mesh.TrianglePrimitives),
                std::move(scene_mesh.DeformData), std::move(scene_mesh.MorphData)
            );
            const auto [me, _] = ::AddMesh(R, ctx.Meshes, SceneEntity, std::move(mesh), std::nullopt);
            mesh_entity = me;
            R.emplace<Path>(mesh_entity, source_path);
            R.emplace<SourceMeshIndex>(mesh_entity, mi);
            R.emplace<SourceMeshKind>(mesh_entity, MeshKind::Triangles);
            R.emplace<MeshSourceLayout>(mesh_entity, std::move(layout));
            if (!scene_mesh.Name.empty()) R.emplace<MeshName>(mesh_entity, MeshName{scene_mesh.Name});
            if (mesh_pbr_mask != 0) R.emplace<PbrMeshFeatures>(mesh_entity, mesh_pbr_mask);
            mesh_morphs.emplace_back(std::move(morph_data_copy));
        } else {
            mesh_morphs.emplace_back(std::nullopt);
        }
        if (first_mesh_entity == entt::null && mesh_entity != entt::null) first_mesh_entity = mesh_entity;
        mesh_entities.emplace_back(mesh_entity);

        auto create_extra = [&](std::optional<::MeshData> &data, ::MeshVertexAttributes &attrs, MeshKind kind) -> entt::entity {
            if (!data) return entt::null;
            auto m = ctx.Meshes.CreateMesh(std::move(*data), std::move(attrs), {});
            const auto [e, _] = ::AddMesh(R, ctx.Meshes, SceneEntity, std::move(m), std::nullopt);
            R.emplace<Path>(e, source_path);
            R.emplace<SourceMeshIndex>(e, mi);
            R.emplace<SourceMeshKind>(e, kind);
            if (!scene_mesh.Name.empty()) R.emplace<MeshName>(e, MeshName{scene_mesh.Name});
            if (first_mesh_entity == entt::null) first_mesh_entity = e;
            return e;
        };
        auto lines_entity = create_extra(scene_mesh.Lines, scene_mesh.LineAttrs, MeshKind::Lines);
        auto points_entity = create_extra(scene_mesh.Points, scene_mesh.PointAttrs, MeshKind::Points);
        extra_entities_per_mesh[mi] = {lines_entity, points_entity};
    }

    const auto name_prefix = source_path.stem().string();
    R.patch<NameRegistry>(SceneEntity, [&](auto &registry) {
        registry.Names.reserve(registry.Names.size() + source.Objects.size());
    });
    std::unordered_map<uint32_t, entt::entity> object_entities_by_node;
    object_entities_by_node.reserve(source.Objects.size());
    std::unordered_map<uint32_t, std::vector<entt::entity>> skinned_mesh_instances_by_skin;
    skinned_mesh_instances_by_skin.reserve(source.Skins.size());
    std::unordered_map<uint32_t, entt::entity> armature_data_entities_by_skin;
    armature_data_entities_by_skin.reserve(source.Skins.size());

    std::vector<entt::entity> all_imported_objects;
    entt::entity first_object_entity = entt::null,
                 first_mesh_object_entity = entt::null,
                 first_camera_object_entity = entt::null,
                 first_root_empty_entity = entt::null,
                 first_armature_entity = entt::null;
    for (uint32_t i = 0; i < source.Objects.size(); ++i) {
        const auto &object = source.Objects[i];
        const auto object_name = object.Name.empty() ? std::format("{}_{}", name_prefix, i) : object.Name;
        entt::entity object_entity = entt::null;
        // Prefer Triangles, then Lines, then Points (for Lines/Points-only source meshes).
        const auto primary_mesh_entity = [&]() -> entt::entity {
            if (object.ObjectType != gltf::Object::Type::Mesh || !object.MeshIndex) return entt::null;
            const auto mi = *object.MeshIndex;
            if (mi < mesh_entities.size() && mesh_entities[mi] != entt::null) return mesh_entities[mi];
            if (mi < extra_entities_per_mesh.size()) {
                const auto &[lines, points] = extra_entities_per_mesh[mi];
                return lines != entt::null ? lines : points;
            }
            return entt::null;
        }();
        if (primary_mesh_entity != entt::null) {
            object_entity = ::AddMeshInstance(
                R, SceneEntity,
                primary_mesh_entity,
                {.Name = object_name, .Transform = object.WorldTransform, .Select = MeshInstanceCreateInfo::SelectBehavior::None, .Visible = true}
            );
        } else if (object.ObjectType == gltf::Object::Type::Camera && object.CameraIndex && *object.CameraIndex < source.Cameras.size()) {
            object_entity = ::AddCamera(R, ctx.Meshes, ctx.Buffers, SceneEntity, {.Name = object_name, .Transform = object.WorldTransform, .Select = MeshInstanceCreateInfo::SelectBehavior::None});
            const auto &scd = source.Cameras[*object.CameraIndex];
            R.replace<::Camera>(object_entity, scd.Camera);
            R.emplace<SourceCameraIndex>(object_entity, *object.CameraIndex);
            if (!scd.Name.empty()) R.emplace<CameraName>(object_entity, scd.Name);
        } else if (object.ObjectType == gltf::Object::Type::Light && object.LightIndex && *object.LightIndex < source.Lights.size()) {
            const auto &sld = source.Lights[*object.LightIndex];
            object_entity = ::AddLight(R, ctx.Meshes, ctx.Buffers, SceneEntity, {.Name = object_name, .Transform = object.WorldTransform, .Select = MeshInstanceCreateInfo::SelectBehavior::None}, sld.Light);
            R.emplace<SourceLightIndex>(object_entity, *object.LightIndex);
            if (!sld.Name.empty()) R.emplace<LightName>(object_entity, sld.Name);
        } else {
            object_entity = ::AddEmpty(R, ctx.Meshes, ctx.Buffers, SceneEntity, {.Name = object_name, .Transform = object.WorldTransform, .Select = MeshInstanceCreateInfo::SelectBehavior::None});
        }
        // Companion instances for non-triangle primitives, skipping whichever already serves as primary.
        if (object.ObjectType == gltf::Object::Type::Mesh && object.MeshIndex && *object.MeshIndex < extra_entities_per_mesh.size()) {
            const auto &extras = extra_entities_per_mesh[*object.MeshIndex];
            for (const auto extra_entity : {extras.Lines, extras.Points}) {
                if (extra_entity == entt::null || extra_entity == primary_mesh_entity) continue;
                const auto extra_instance = ::AddMeshInstance(
                    R, SceneEntity,
                    extra_entity,
                    {.Name = object_name, .Transform = object.WorldTransform, .Select = MeshInstanceCreateInfo::SelectBehavior::None, .Visible = true}
                );
                // Keep line/point primitive instances in lockstep with the primary glTF object transform.
                SetParent(R, extra_instance, object_entity);
            }
        }

        object_entities_by_node[object.NodeIndex] = object_entity;
        R.emplace<SourceNodeIndex>(object_entity, object.NodeIndex);
        // `object.Name` was already synthesized by the loader's MakeNodeName when source was
        // empty; `source.Nodes` preserves the raw value. Capture source-empty / collision-renamed.
        if (object.NodeIndex < source.Nodes.size()) {
            const auto &raw_name = source.Nodes[object.NodeIndex].Name;
            if (raw_name.empty()) R.emplace<SourceEmptyName>(object_entity);
            else if (const auto *n = R.try_get<const Name>(object_entity); n && n->Value != raw_name) {
                R.emplace<SourceObjectName>(object_entity, SourceObjectName{raw_name});
            }
        }
        all_imported_objects.emplace_back(object_entity);
        // glTF node.skin is deform linkage, not a transform-parent relationship.
        if (object.SkinIndex && R.all_of<Instance>(object_entity)) skinned_mesh_instances_by_skin[*object.SkinIndex].emplace_back(object_entity);
        if (first_object_entity == entt::null) first_object_entity = object_entity;
        if (first_mesh_object_entity == entt::null && object.ObjectType == gltf::Object::Type::Mesh) first_mesh_object_entity = object_entity;
        if (first_camera_object_entity == entt::null && object.ObjectType == gltf::Object::Type::Camera) first_camera_object_entity = object_entity;
        if (first_root_empty_entity == entt::null && object.ObjectType == gltf::Object::Type::Empty && !object.ParentNodeIndex) first_root_empty_entity = object_entity;
    }

    for (const auto &object : source.Objects) {
        if (!object.ParentNodeIndex) continue;

        const auto child_it = object_entities_by_node.find(object.NodeIndex);
        if (child_it == object_entities_by_node.end()) continue;
        const auto parent_it = object_entities_by_node.find(*object.ParentNodeIndex);
        if (parent_it == object_entities_by_node.end()) continue;
        SetParent(R, child_it->second, parent_it->second);
    }

    std::unordered_map<uint32_t, const gltf::Node *> scene_nodes_by_index;
    scene_nodes_by_index.reserve(source.Nodes.size());
    for (const auto &node : source.Nodes) scene_nodes_by_index.emplace(node.NodeIndex, &node);

    // Stubs for out-of-scene nodes (referenced only by non-default scenes) so build emits them
    // like the file round-trip does. They carry only what build needs and aren't in
    // `object_entities_by_node`, so runtime systems that walk the scene tree don't see them.
    for (const auto &node : source.Nodes) {
        if (node.InScene) continue;
        const auto e = R.create();
        R.emplace<SourceNodeIndex>(e, node.NodeIndex);
        R.emplace<Transform>(e, node.LocalTransform);
        R.emplace<WorldTransform>(e, node.WorldTransform);
        if (node.MeshIndex && *node.MeshIndex < mesh_entities.size() && mesh_entities[*node.MeshIndex] != entt::null) {
            R.emplace<Instance>(e, mesh_entities[*node.MeshIndex]);
        }
        if (node.Name.empty()) R.emplace<SourceEmptyName>(e);
        else R.emplace<Name>(e, node.Name);
    }

    // KHR_physics_rigid_bodies: promote loader's index-keyed resource arrays to registry entities,
    // each tagged with its source index for round-trip ordering.
    {
        const auto promote = [&]<typename TComp, typename TIndex>(const auto &src) {
            std::vector<entt::entity> entities;
            entities.reserve(src.size());
            for (uint32_t i = 0; i < src.size(); ++i) {
                const auto e = R.create();
                R.emplace<TComp>(e, src[i]);
                R.emplace<TIndex>(e, i);
                entities.emplace_back(e);
            }
            return entities;
        };
        const auto material_entities = promote.operator()<PhysicsMaterial, SourcePhysicsMaterialIndex>(source.PhysicsMaterials);
        const auto jointdef_entities = promote.operator()<::PhysicsJointDef, SourcePhysicsJointDefIndex>(source.PhysicsJointDefs);

        // Dedupe system names across all filters into CollisionSystem entities.
        std::unordered_map<std::string, entt::entity> system_entity_by_name;
        const auto resolve_systems = [&](const std::vector<std::string> &names) {
            std::vector<entt::entity> out;
            out.reserve(names.size());
            for (const auto &n : names) {
                auto [it, inserted] = system_entity_by_name.try_emplace(n, entt::null);
                if (inserted) {
                    it->second = R.create();
                    R.emplace<CollisionSystem>(it->second, CollisionSystem{.Name = n});
                }
                out.emplace_back(it->second);
            }
            return out;
        };

        std::vector<entt::entity> filter_entities;
        filter_entities.reserve(source.CollisionFilters.size());
        for (uint32_t i = 0; i < source.CollisionFilters.size(); ++i) {
            const auto &f = source.CollisionFilters[i];
            CollisionFilter filter{.Systems = resolve_systems(f.CollisionSystems), .Name = f.Name};
            // KHR schema forbids both collideWith and notCollideWith; prefer allowlist if both appear.
            if (!f.CollideWithSystems.empty()) {
                filter.Mode = CollideMode::Allowlist;
                filter.CollideSystems = resolve_systems(f.CollideWithSystems);
            } else if (!f.NotCollideWithSystems.empty()) {
                filter.Mode = CollideMode::Blocklist;
                filter.CollideSystems = resolve_systems(f.NotCollideWithSystems);
            }
            const auto e = R.create();
            R.emplace<CollisionFilter>(e, std::move(filter));
            R.emplace<SourceCollisionFilterIndex>(e, i);
            filter_entities.emplace_back(e);
        }

        auto resolve_mat = [&](std::optional<uint32_t> idx) {
            return idx && *idx < material_entities.size() ? material_entities[*idx] : null_entity;
        };
        auto resolve_filter = [&](std::optional<uint32_t> idx) {
            return idx && *idx < filter_entities.size() ? filter_entities[*idx] : null_entity;
        };

        for (const auto &node : source.Nodes) {
            auto it = object_entities_by_node.find(node.NodeIndex);
            if (it == object_entities_by_node.end()) continue;
            const auto entity = it->second;

            if (node.Collider) {
                auto collider = *node.Collider;
                if (IsMeshBackedShape(collider.Shape)) {
                    if (node.ColliderGeometryMeshIndex && *node.ColliderGeometryMeshIndex < mesh_entities.size()) {
                        collider.MeshEntity = mesh_entities[*node.ColliderGeometryMeshIndex];
                    } else if (R.all_of<Instance>(entity)) {
                        collider.MeshEntity = R.get<const Instance>(entity).Entity;
                    }
                }
                R.emplace<ColliderShape>(entity, std::move(collider));
                // Imported collider state is authoritative — engine must not auto-derive over it.
                R.emplace<ColliderPolicy>(entity, ColliderPolicy{.AutoFitDims = false, .LockedKind = true});
                if (node.Material) {
                    R.replace<ColliderMaterial>(entity, ColliderMaterial{
                                                            .PhysicsMaterialEntity = resolve_mat(node.Material->PhysicsMaterialIndex),
                                                            .CollisionFilterEntity = resolve_filter(node.Material->CollisionFilterIndex),
                                                        });
                }
            }
            if (node.Motion) {
                R.emplace<PhysicsMotion>(entity, *node.Motion);
                if (node.Velocity) R.replace<PhysicsVelocity>(entity, *node.Velocity);
            }
            if (node.Trigger) {
                const auto &td = *node.Trigger;
                if (td.Shape) {
                    // GeometryTrigger: reuse ColliderShape + TriggerTag. Skip if a solid collider
                    // already took this entity — KHR declares nodes as one-or-the-other.
                    if (!R.all_of<ColliderShape>(entity)) {
                        ColliderShape shape{.Shape = *td.Shape};
                        if (td.GeometryMeshIndex && *td.GeometryMeshIndex < mesh_entities.size()) {
                            shape.MeshEntity = mesh_entities[*td.GeometryMeshIndex];
                        }
                        R.emplace<ColliderShape>(entity, std::move(shape));
                        R.emplace<ColliderPolicy>(entity, ColliderPolicy{.AutoFitDims = false, .LockedKind = true});
                        R.emplace<TriggerTag>(entity);
                        R.patch<ColliderMaterial>(entity, [&](auto &m) { m.CollisionFilterEntity = resolve_filter(td.CollisionFilterIndex); });
                    }
                } else {
                    // NodesTrigger: compound zone.
                    std::vector<entt::entity> resolved_nodes;
                    resolved_nodes.reserve(td.NodeIndices.size());
                    for (const auto node_idx : td.NodeIndices) {
                        auto nit = object_entities_by_node.find(node_idx);
                        resolved_nodes.emplace_back(nit != object_entities_by_node.end() ? nit->second : entt::null);
                    }
                    R.emplace<TriggerNodes>(entity, TriggerNodes{.Nodes = std::move(resolved_nodes), .CollisionFilterEntity = resolve_filter(td.CollisionFilterIndex)});
                }
            }
            if (node.Joint) {
                const auto &jd = *node.Joint;
                auto nit = object_entities_by_node.find(jd.ConnectedNodeIndex);
                const auto def_entity = jd.JointDefIndex < jointdef_entities.size() ? jointdef_entities[jd.JointDefIndex] : null_entity;
                R.emplace<PhysicsJoint>(
                    entity,
                    PhysicsJoint{.ConnectedNode = nit != object_entities_by_node.end() ? nit->second : entt::null, .JointDefEntity = def_entity, .EnableCollision = jd.EnableCollision}
                );
            }
        }
    }

    for (const auto &skin : source.Skins) {
        const auto armature_data_entity = R.create();
        auto &armature = R.emplace<Armature>(armature_data_entity);
        armature_data_entities_by_skin[skin.SkinIndex] = armature_data_entity;

        ArmatureImportedSkin imported_skin{
            .SkinIndex = skin.SkinIndex,
            .SkeletonNodeIndex = skin.SkeletonNodeIndex,
            .AnchorNodeIndex = skin.AnchorNodeIndex,
            .OrderedJointNodeIndices = {},
            .InverseBindMatrices = skin.InverseBindMatrices,
        };
        imported_skin.OrderedJointNodeIndices.reserve(skin.Joints.size());

        std::unordered_map<uint32_t, BoneId> joint_node_to_bone_id;
        joint_node_to_bone_id.reserve(skin.Joints.size());
        for (const auto &joint : skin.Joints) {
            std::optional<BoneId> parent_bone_id;
            if (joint.ParentJointNodeIndex) {
                if (const auto parent_it = joint_node_to_bone_id.find(*joint.ParentJointNodeIndex);
                    parent_it != joint_node_to_bone_id.end()) {
                    parent_bone_id = parent_it->second;
                }
            }

            const auto joint_name = joint.Name.empty() ? std::format("Joint{}", joint.JointNodeIndex) : joint.Name;
            const auto bone_id = armature.AddBone(joint_name, parent_bone_id, joint.RestLocal, joint.JointNodeIndex);
            joint_node_to_bone_id.emplace(joint.JointNodeIndex, bone_id);
            if (const auto object_it = object_entities_by_node.find(joint.JointNodeIndex);
                object_it != object_entities_by_node.end() &&
                R.all_of<Instance>(object_it->second) &&
                !R.all_of<PhysicsMotion>(object_it->second) &&
                !R.all_of<BoneAttachment>(object_it->second)) {
                R.emplace<BoneAttachment>(object_it->second, armature_data_entity, bone_id);
            }
            imported_skin.OrderedJointNodeIndices.emplace_back(joint.JointNodeIndex);
        }
        imported_skin.InverseBindMatrices.resize(imported_skin.OrderedJointNodeIndices.size(), I4);
        armature.ImportedSkin = std::move(imported_skin);
        armature.FinalizeStructure();
        if (!skin.AnchorNodeIndex) {
            return std::unexpected{std::format("glTF import failed for '{}': skin {} has no deterministic anchor node.", source_path.string(), skin.SkinIndex)};
        }

        const auto anchor_it = scene_nodes_by_index.find(*skin.AnchorNodeIndex);
        if (anchor_it == scene_nodes_by_index.end() || !anchor_it->second->InScene) {
            return std::unexpected{std::format("glTF import failed for '{}': skin {} anchor node {} is not in the imported scene.", source_path.string(), skin.SkinIndex, *skin.AnchorNodeIndex)};
        }

        const auto armature_entity = R.create();
        R.emplace<ObjectKind>(armature_entity, ObjectType::Armature);
        R.emplace<ArmatureObject>(armature_entity, armature_data_entity);
        const auto &t = anchor_it->second->WorldTransform;
        R.emplace<Transform>(armature_entity, t);
        R.emplace<WorldTransform>(armature_entity, t);
        R.emplace<Name>(armature_entity, ::CreateName(R, SceneEntity, skin.Name.empty() ? std::format("{}_Armature{}", name_prefix, skin.SkinIndex) : skin.Name));
        if (skin.Name.empty()) R.emplace<SourceEmptyName>(armature_entity);
        else R.emplace<SkinName>(armature_entity, SkinName{skin.Name});

        if (skin.ParentObjectNodeIndex) {
            if (const auto parent_it = object_entities_by_node.find(*skin.ParentObjectNodeIndex);
                parent_it != object_entities_by_node.end()) {
                SetParent(R, armature_entity, parent_it->second);
            }
        }

        all_imported_objects.emplace_back(armature_entity);
        if (first_armature_entity == entt::null) first_armature_entity = armature_entity;
        if (first_object_entity == entt::null) first_object_entity = armature_entity;

        if (const auto skinned_it = skinned_mesh_instances_by_skin.find(skin.SkinIndex);
            skinned_it != skinned_mesh_instances_by_skin.end()) {
            for (const auto mesh_instance_entity : skinned_it->second) {
                if (!R.valid(mesh_instance_entity) || !R.all_of<Instance>(mesh_instance_entity)) continue;
                R.emplace_or_replace<ArmatureModifier>(mesh_instance_entity, armature_data_entity, armature_entity);
                SetParent(R, mesh_instance_entity, armature_entity);
            }
        } else {
            return std::unexpected{std::format("glTF import failed '{}': skin {} is used but no mesh instances were emitted for skin binding.", source_path.string(), skin.SkinIndex)};
        }

        { // Create pose state — GPU deform buffer allocation deferred to ProcessComponentEvents.
            ArmaturePoseState pose_state;
            pose_state.BonePoseDelta.resize(armature.Bones.size(), Transform{});
            pose_state.BoneUserOffset.resize(armature.Bones.size(), Transform{});
            pose_state.BonePoseWorld.resize(armature.Bones.size(), I4);
            R.emplace<ArmaturePoseState>(armature_data_entity, std::move(pose_state));
        }
        ::CreateBoneInstances(R, ctx.Meshes, SceneEntity, armature_entity, armature_data_entity);
        // Mark each bone entity with its source joint NodeIndex (for SaveScene round-trip).
        const auto &bone_entities_for_source = R.get<const ArmatureObject>(armature_entity).BoneEntities;
        for (uint32_t i = 0; i < armature.Bones.size(); ++i) {
            const auto joint_node_index = armature.Bones[i].JointNodeIndex;
            if (!joint_node_index) continue;
            R.emplace<SourceNodeIndex>(bone_entities_for_source[i], *joint_node_index);
            if (*joint_node_index < source.Nodes.size() && source.Nodes[*joint_node_index].Name.empty()) {
                R.emplace<SourceEmptyName>(bone_entities_for_source[i]);
            }
        }

        // Auto-wire Child Of on bones whose joint node has a physics-driven ancestor object,
        // so the skin follows simulated motion when an asset pairs rigid bodies with skinning via the scene graph.
        // Target is the nearest ancestor object with PhysicsMotion; InverseMatrix bakes the rest offset.
        {
            const auto find_physics_ancestor_entity = [&](uint32_t node_index) -> entt::entity {
                for (std::optional<uint32_t> cur = node_index; cur;) {
                    if (const auto oit = object_entities_by_node.find(*cur);
                        oit != object_entities_by_node.end() && R.all_of<PhysicsMotion>(oit->second)) return oit->second;
                    const auto nit = scene_nodes_by_index.find(*cur);
                    if (nit == scene_nodes_by_index.end()) break;
                    cur = nit->second->ParentNodeIndex;
                }
                return entt::null;
            };
            const auto &arm_obj = R.get<const ArmatureObject>(armature_entity);
            const mat4 armature_world = ToMatrix(R.get<const WorldTransform>(armature_entity));
            for (uint32_t i = 0; i < armature.Bones.size(); ++i) {
                const auto &bone = armature.Bones[i];
                if (!bone.JointNodeIndex) continue;
                const auto target = find_physics_ancestor_entity(*bone.JointNodeIndex);
                if (target == entt::null) continue;
                R.emplace<BoneConstraints>(
                    arm_obj.BoneEntities[i],
                    BoneConstraints{
                        .Stack = {BoneConstraint{
                            .TargetEntity = target,
                            .Influence = 1.f,
                            .Data = ChildOfData{.InverseMatrix = glm::inverse(ToMatrix(R.get<const WorldTransform>(target))) * (armature_world * bone.RestWorld)},
                        }}
                    }
                );
            }
        }
    }

    // Per source-derived entity: tag with source parent / sibling position / matrix-form flag,
    // and stash source `LocalTransform` for lossless save (see `SourceLocalTransform`).
    // `Transform`/`WorldTransform`/`ParentInverse` are left as set by AddMeshInstance/etc. + SetParent
    // (Transform == world, ParentInverse == inverse(parent_world)).
    for (const auto [entity, sni] : R.view<const SourceNodeIndex>().each()) {
        const auto it = scene_nodes_by_index.find(sni.Value);
        if (it == scene_nodes_by_index.end()) continue;
        const auto &source_node = *it->second;
        if (source_node.ParentNodeIndex) {
            R.emplace<SourceParentNodeIndex>(entity, *source_node.ParentNodeIndex);
            if (const auto parent_it = scene_nodes_by_index.find(*source_node.ParentNodeIndex); parent_it != scene_nodes_by_index.end()) {
                const auto &siblings = parent_it->second->ChildrenNodeIndices;
                if (const auto pos = std::ranges::find(siblings, sni.Value); pos != siblings.end()) {
                    R.emplace<SourceSiblingIndex>(entity, uint32_t(pos - siblings.begin()));
                }
            }
        }
        if (source_node.SourceMatrix) R.emplace<SourceMatrixTransform>(entity, *source_node.SourceMatrix);
        else R.emplace<SourceLocalTransform>(entity, source_node.LocalTransform);
    }

    std::unordered_set<uint32_t> joint_node_indices;
    for (const auto &skin : source.Skins) {
        for (const auto &joint : skin.Joints) joint_node_indices.emplace(joint.JointNodeIndex);
    }
    std::unordered_map<uint32_t, std::vector<std::pair<entt::entity, BoneId>>> armature_targets_by_joint_node;
    for (const auto &entry : armature_data_entities_by_skin) {
        const auto armature_data_entity = entry.second;
        const auto &armature = R.get<const Armature>(armature_data_entity);
        for (const auto &bone : armature.Bones) {
            if (!bone.JointNodeIndex) continue;
            armature_targets_by_joint_node[*bone.JointNodeIndex].emplace_back(armature_data_entity, bone.Id);
        }
    }

    // Set up morph weight state for mesh instances with morph targets.
    // GPU allocation deferred to ProcessComponentEvents (GpuWeightRange left as default {0,0}).
    // Build a map: node_index -> mesh instance entity, for resolving weight animation channels.
    std::unordered_map<uint32_t, entt::entity> morph_instance_by_node;
    for (const auto &object : source.Objects) {
        if (object.ObjectType != gltf::Object::Type::Mesh || !object.MeshIndex) continue;
        if (*object.MeshIndex >= mesh_morphs.size() || !mesh_morphs[*object.MeshIndex]) continue;
        const auto obj_it = object_entities_by_node.find(object.NodeIndex);
        if (obj_it == object_entities_by_node.end()) continue;
        const auto instance_entity = obj_it->second;
        if (!R.all_of<Instance>(instance_entity)) continue;

        const auto &morph = *mesh_morphs[*object.MeshIndex];
        if (morph.TargetCount == 0) continue;

        MorphWeightState state;
        if (object.NodeWeights) {
            state.Weights.resize(morph.TargetCount, 0.f);
            std::copy_n(object.NodeWeights->begin(), std::min(uint32_t(object.NodeWeights->size()), morph.TargetCount), state.Weights.begin());
        } else {
            state.Weights = morph.DefaultWeights;
        }
        R.emplace<MorphWeightState>(instance_entity, std::move(state));
        morph_instance_by_node[object.NodeIndex] = instance_entity;
    }

    // Resolve object/node transform animations (empties, meshes, cameras, lights).
    // Channels targeting skin joints are handled by ArmatureAnimation and skipped here.
    std::unordered_map<entt::entity, std::pair<Transform, Transform>> node_anim_bindings;
    node_anim_bindings.reserve(object_entities_by_node.size());
    for (const auto &[node_index, object_entity] : object_entities_by_node) {
        if (!R.valid(object_entity)) continue;
        const auto node_it = scene_nodes_by_index.find(node_index);
        if (node_it == scene_nodes_by_index.end()) continue;

        Transform parent_bind_world{};
        if (const auto parent_node_index = node_it->second->ParentNodeIndex) {
            if (const auto parent_it = scene_nodes_by_index.find(*parent_node_index); parent_it != scene_nodes_by_index.end()) {
                parent_bind_world = parent_it->second->WorldTransform;
            }
        }
        node_anim_bindings.emplace(object_entity, std::pair{node_it->second->LocalTransform, parent_bind_world});
    }

    bool imported_animation = false;
    const auto append_armature_clip = [&](entt::entity target_data_entity, ::AnimationClip &&resolved_clip) {
        if (resolved_clip.Channels.empty()) return;
        imported_animation = true;
        if (auto *existing = R.try_get<ArmatureAnimation>(target_data_entity)) {
            existing->Clips.emplace_back(std::move(resolved_clip));
        } else {
            R.emplace<ArmatureAnimation>(target_data_entity, ArmatureAnimation{.Clips = {std::move(resolved_clip)}});
        }
    };
    const auto append_morph_clip = [&](entt::entity instance_entity, MorphWeightClip &&resolved_clip) {
        if (resolved_clip.Channels.empty()) return;
        imported_animation = true;
        if (auto *existing = R.try_get<MorphWeightAnimation>(instance_entity)) {
            existing->Clips.emplace_back(std::move(resolved_clip));
        } else {
            R.emplace<MorphWeightAnimation>(instance_entity, MorphWeightAnimation{.Clips = {std::move(resolved_clip)}});
        }
    };
    const auto append_node_clip = [&](entt::entity object_entity, ::AnimationClip &&resolved_clip) {
        if (resolved_clip.Channels.empty()) return;
        imported_animation = true;
        if (auto *existing = R.try_get<NodeTransformAnimation>(object_entity)) {
            existing->Clips.emplace_back(std::move(resolved_clip));
            return;
        }
        const auto binding_it = node_anim_bindings.find(object_entity);
        if (binding_it == node_anim_bindings.end()) return;
        R.emplace<NodeTransformAnimation>(
            object_entity,
            NodeTransformAnimation{.Clips = {std::move(resolved_clip)}, .ActiveClipIndex = 0, .RestLocal = binding_it->second.first, .ParentBindWorld = binding_it->second.second}
        );
    };

    // Resolve armature, morph-weight, and node TRS channels in one pass per source clip.
    for (const auto &anim_clip : source.Animations) {
        std::unordered_map<entt::entity, ::AnimationClip> armature_clips_by_entity;
        std::unordered_map<entt::entity, MorphWeightClip> morph_clips_by_entity;
        std::unordered_map<entt::entity, ::AnimationClip> node_clips_by_entity;
        for (const auto &ch : anim_clip.Channels) {
            if (ch.Target == AnimationPath::Weights) {
                const auto inst_it = morph_instance_by_node.find(ch.TargetNodeIndex);
                if (inst_it == morph_instance_by_node.end()) continue;
                auto &resolved_clip = morph_clips_by_entity
                                          .try_emplace(inst_it->second, MorphWeightClip{.Name = anim_clip.Name, .DurationSeconds = anim_clip.DurationSeconds, .Channels = {}})
                                          .first->second;
                resolved_clip.Channels.emplace_back(MorphWeightChannel{
                    .Interp = ch.Interp,
                    .TimesSeconds = ch.TimesSeconds,
                    .Values = ch.Values,
                });
                continue;
            }

            if (const auto armature_it = armature_targets_by_joint_node.find(ch.TargetNodeIndex);
                armature_it != armature_targets_by_joint_node.end()) {
                for (const auto &[target_data_entity, bone_id] : armature_it->second) {
                    const auto &armature = R.get<const Armature>(target_data_entity);
                    const auto bone_index = armature.FindBoneIndex(bone_id).value_or(InvalidBoneIndex);
                    auto &resolved_clip = armature_clips_by_entity
                                              .try_emplace(target_data_entity, ::AnimationClip{.Name = anim_clip.Name, .DurationSeconds = anim_clip.DurationSeconds, .Channels = {}})
                                              .first->second;
                    resolved_clip.Channels.emplace_back(::AnimationChannel{.BoneIndex = bone_index, .TargetBoneId = bone_id, .Target = ch.Target, .Interp = ch.Interp, .TimesSeconds = ch.TimesSeconds, .Values = ch.Values});
                }
                continue;
            }

            if (joint_node_indices.contains(ch.TargetNodeIndex)) continue;

            const auto object_it = object_entities_by_node.find(ch.TargetNodeIndex);
            if (object_it == object_entities_by_node.end() || !R.valid(object_it->second)) continue;

            auto &resolved_clip = node_clips_by_entity
                                      .try_emplace(object_it->second, ::AnimationClip{.Name = anim_clip.Name, .DurationSeconds = anim_clip.DurationSeconds, .Channels = {}})
                                      .first->second;
            resolved_clip.Channels.emplace_back(::AnimationChannel{.BoneIndex = 0, .Target = ch.Target, .Interp = ch.Interp, .TimesSeconds = ch.TimesSeconds, .Values = ch.Values});
        }

        for (auto &[target_data_entity, resolved_clip] : armature_clips_by_entity) append_armature_clip(target_data_entity, std::move(resolved_clip));
        for (auto &[instance_entity, resolved_clip] : morph_clips_by_entity) append_morph_clip(instance_entity, std::move(resolved_clip));
        for (auto &[object_entity, resolved_clip] : node_clips_by_entity) append_node_clip(object_entity, std::move(resolved_clip));
    }

    { // Get timeline range from imported animation durations
        float max_dur = 0;
        for (const auto [_, anim] : R.view<const ArmatureAnimation>().each()) {
            for (const auto &clip : anim.Clips) max_dur = std::max(max_dur, clip.DurationSeconds);
        }
        for (const auto [_, anim] : R.view<const MorphWeightAnimation>().each()) {
            for (const auto &clip : anim.Clips) max_dur = std::max(max_dur, clip.DurationSeconds);
        }
        for (const auto [_, anim] : R.view<const NodeTransformAnimation>().each()) {
            for (const auto &clip : anim.Clips) max_dur = std::max(max_dur, clip.DurationSeconds);
        }
        if (max_dur > 0) R.patch<AnimationTimeline>(SceneEntity, [&](auto &tl) { tl.EndFrame = int(std::ceil(max_dur * tl.Fps)); });
    }

    if (source.ImageBasedLight) {
        auto ibl_batch = BeginTextureUploadBatch(ctx.Vk.Device, ctx.CommandPool, ctx.Buffers.Ctx);
        auto scene_world = CreateIblFromExtIbl(ctx.Vk, ibl_batch, ctx.Slots, source.Images, *source.ImageBasedLight);
        SubmitTextureUploadBatch(ibl_batch, ctx.Vk.Queue, ctx.OneShotFence, ctx.Vk.Device);
        if (!scene_world) {
            std::cerr << std::format("Warning: Failed to import EXT_lights_image_based scene world from '{}': {}\n", source_path.string(), scene_world.error());
        } else {
            auto &environments = ctx.Environments;
            if (environments.ImportedSceneWorld) {
                ReleaseCubeSamplerSlot(ctx.Slots, environments.ImportedSceneWorld->DiffuseEnv.SamplerSlot);
                ReleaseCubeSamplerSlot(ctx.Slots, environments.ImportedSceneWorld->SpecularEnv.SamplerSlot);
            }
            environments.ImportedSceneWorld = std::move(*scene_world);
            environments.SceneWorldRotation = glm::mat3_cast(source.ImageBasedLight->Rotation);
            const auto &pre = *environments.ImportedSceneWorld;
            environments.SceneWorld = {.Ibl = MakeIblSamplers(pre, environments), .Name = pre.Name};
        }
    }

    const auto active_entity =
        first_camera_object_entity != entt::null ? first_camera_object_entity :
        first_mesh_object_entity != entt::null   ? first_mesh_object_entity :
        first_armature_entity != entt::null      ? first_armature_entity :
        first_root_empty_entity != entt::null    ? first_root_empty_entity :
                                                   first_object_entity;
    R.clear<Active, Selected>();
    if (active_entity != entt::null) R.emplace<Active>(active_entity);
    for (const auto e : all_imported_objects) R.emplace<Selected>(e);
    import_rollback_guard.Enabled = false;

    return gltf::PopulateResult{
        .FirstMesh = first_mesh_entity,
        .Active = active_entity,
        .FirstCameraObject = first_camera_object_entity,
        .ImportedAnimation = imported_animation,
    };
}

gltf::Scene gltf::BuildGltfScene(const entt::registry &r, entt::entity scene_entity, const SceneBuffers &buffers, const MeshStore &meshes) {
    gltf::Scene scene;

    // Order entities in `view` by their `TIndex` sidecar value. Entities without `TIndex`
    // (runtime-added) land after the source range. Used for cameras, lights, physics resources.
    const auto ordered_by_source = [&]<typename TIndex>(auto view) {
        std::vector<std::pair<uint32_t, entt::entity>> ordered;
        uint32_t next = 0;
        for (const auto e : view) {
            if (const auto *si = r.try_get<const TIndex>(e)) {
                ordered.emplace_back(si->Value, e);
                next = std::max(next, si->Value + 1u);
            }
        }
        for (const auto e : view) {
            if (!r.all_of<TIndex>(e)) ordered.emplace_back(next++, e);
        }
        std::ranges::sort(ordered, {}, &std::pair<uint32_t, entt::entity>::first);
        return ordered;
    };

    // Source-form scene metadata + texture/image/sampler arrays come from the GltfSourceAssets
    // sidecar — encoded image bytes, sampler-config collapse, and asset.* metadata aren't
    // recoverable from registry/GPU state. Cameras and lights emit below from per-entity
    // components (see CameraName/LightName). Materials reconstruct via FromGpu(GPU buffer) +
    // patch from the per-material delta in `MaterialMetas`.
    const auto *src_assets = r.try_get<const GltfSourceAssets>(scene_entity);
    if (src_assets) {
        scene.Copyright = src_assets->Copyright;
        scene.Generator = src_assets->Generator;
        scene.MinVersion = src_assets->MinVersion;
        scene.AssetExtras = src_assets->AssetExtras;
        scene.AssetExtensions = src_assets->AssetExtensions;
        scene.DefaultSceneName = src_assets->DefaultSceneName;
        scene.DefaultSceneRoots = src_assets->DefaultSceneRoots;
        scene.ExtensionsRequired = src_assets->ExtensionsRequired;
        scene.MaterialVariants = src_assets->MaterialVariants;
        scene.ExtrasByEntity = src_assets->ExtrasByEntity;
        scene.Textures = src_assets->Textures;
        scene.Images = src_assets->Images;
        scene.Samplers = src_assets->Samplers;
        scene.ImageBasedLight = src_assets->ImageBasedLight;
    }
    // Materials: skip the engine "Default" at registry index 0; loaded gltf materials live at
    // [1, count). Reconstruct each via `FromGpu` then patch with `MaterialSourceMeta` to restore
    // KHR_materials_emissive_strength split, KHR_texture_transform meta, source texture indices,
    // and the optionality of extension blocks (FromGpu's value-vs-default gate is lossy for them).
    const auto &names = r.get<const MaterialStore>(scene_entity).Names;
    const auto material_count = buffers.Materials.Count();
    const auto &material_metas = src_assets ? src_assets->MaterialMetas : std::vector<MaterialSourceMeta>{};
    if (material_count > 1) {
        using M = MaterialSourceMeta;
        // Restore an extension's optional<>: bit set + FromGpu produced nullopt (all-defaults block) → default-construct.
        // Bit clear → drop FromGpu's reconstruction (source had no extension).
        const auto sync_ext = [&]<typename T>(std::optional<T> &slot, uint16_t bits, uint16_t bit) {
            if (bits & bit) {
                if (!slot) slot = T{};
            } else slot.reset();
        };
        scene.Materials.reserve(material_count - 1);
        for (uint32_t i = 1; i < material_count; ++i) {
            const auto source_idx = i - 1;
            auto data = gltf::FromGpu(buffers.Materials.Get(i));
            std::string name = i < names.size() ? names[i] : std::string{};
            if (source_idx < material_metas.size()) {
                const auto &meta = material_metas[source_idx];
                const auto bits = meta.ExtensionPresence;
                // Base texture slots + KHR_texture_transform meta.
                const std::array<TextureInfo *, 5> base_tex{&data.BaseColorTexture, &data.MetallicRoughnessTexture, &data.NormalTexture, &data.OcclusionTexture, &data.EmissiveTexture};
                const std::array<gltf::TextureTransformMeta *, 5> base_meta{&data.BaseColorMeta, &data.MetallicRoughnessMeta, &data.NormalMeta, &data.OcclusionMeta, &data.EmissiveMeta};
                for (uint8_t k = 0; k < 5; ++k) {
                    base_tex[k]->Slot = meta.TextureSlots[k];
                    *base_meta[k] = meta.BaseSlotMeta[k];
                }
                // Extension presence — Ior/Dispersion are scalars; the rest are sub-structs.
                if (bits & M::ExtIor) {
                    if (!data.Ior) data.Ior = 1.5f;
                } else data.Ior.reset();
                if (bits & M::ExtDispersion) {
                    if (!data.Dispersion) data.Dispersion = 0.f;
                } else data.Dispersion.reset();
                sync_ext(data.Sheen, bits, M::ExtSheen);
                sync_ext(data.Specular, bits, M::ExtSpecular);
                sync_ext(data.Transmission, bits, M::ExtTransmission);
                sync_ext(data.DiffuseTransmission, bits, M::ExtDiffuseTransmission);
                sync_ext(data.Volume, bits, M::ExtVolume);
                sync_ext(data.Clearcoat, bits, M::ExtClearcoat);
                sync_ext(data.Anisotropy, bits, M::ExtAnisotropy);
                sync_ext(data.Iridescence, bits, M::ExtIridescence);
                // Nested extension texture slots — assigns are no-ops when the optional is empty.
                const auto put = [&](auto &opt, auto field, MaterialTextureSlot s) {
                    if (opt) ((*opt).*field).Slot = meta.TextureSlots[s];
                };
                put(data.Specular, &::Specular::Texture, MTS_Specular);
                put(data.Specular, &::Specular::ColorTexture, MTS_SpecularColor);
                put(data.Sheen, &::Sheen::ColorTexture, MTS_SheenColor);
                put(data.Sheen, &::Sheen::RoughnessTexture, MTS_SheenRoughness);
                put(data.Transmission, &::Transmission::Texture, MTS_Transmission);
                put(data.DiffuseTransmission, &::DiffuseTransmission::Texture, MTS_DiffuseTransmission);
                put(data.DiffuseTransmission, &::DiffuseTransmission::ColorTexture, MTS_DiffuseTransmissionColor);
                put(data.Volume, &::Volume::ThicknessTexture, MTS_VolumeThickness);
                put(data.Clearcoat, &::Clearcoat::Texture, MTS_Clearcoat);
                put(data.Clearcoat, &::Clearcoat::RoughnessTexture, MTS_ClearcoatRoughness);
                put(data.Clearcoat, &::Clearcoat::NormalTexture, MTS_ClearcoatNormal);
                put(data.Anisotropy, &::Anisotropy::Texture, MTS_Anisotropy);
                put(data.Iridescence, &::Iridescence::Texture, MTS_Iridescence);
                put(data.Iridescence, &::Iridescence::ThicknessTexture, MTS_IridescenceThickness);
                // ToGpu folded `EmissiveFactor *= strength`; un-fold for round-trip.
                if (meta.EmissiveStrength) {
                    const float s = *meta.EmissiveStrength;
                    if (s != 0.f) data.EmissiveFactor /= s;
                    data.EmissiveStrength = s;
                }
                if (meta.NameWasEmpty) name.clear();
            }
            scene.Materials.emplace_back(gltf::NamedMaterial{.Value = std::move(data), .Name = std::move(name)});
        }
    }

    // Mesh entities → MeshIndex. Source meshes use SourceMeshIndex for stable round-trip ordering;
    // engine-generated meshes (no SourceMeshIndex) are skipped — they don't belong in scene.Meshes.
    std::unordered_map<entt::entity, uint32_t> mesh_entity_to_index;
    uint32_t source_mesh_count = 0;
    auto source_mesh_view = r.view<const Mesh, const SourceMeshIndex>();
    for (const auto e : source_mesh_view) {
        const auto smi_value = source_mesh_view.get<const SourceMeshIndex>(e).Value;
        mesh_entity_to_index[e] = smi_value;
        source_mesh_count = std::max(source_mesh_count, smi_value + 1u);
    }

    // Source meshes → gltf::MeshData. Triangles populate the .Triangles slot below, Lines and
    // Points entities (which share a SourceMeshIndex with their sibling kinds) populate the
    // .Lines / .Points slots in the second pass.
    scene.Meshes.resize(source_mesh_count);
    for (const auto [entity, smi, layout] : r.view<const SourceMeshIndex, const MeshSourceLayout>().each()) {
        const auto *mesh_ptr = r.try_get<const Mesh>(entity);
        if (!mesh_ptr) continue;
        const auto &mesh = *mesh_ptr;
        const auto store_id = mesh.GetStoreId();
        const auto vertices = meshes.GetVertices(store_id);
        const auto vertex_count = uint32_t(vertices.size());

        ::MeshData triangles;
        triangles.Positions.reserve(vertex_count);
        for (const auto &v : vertices) triangles.Positions.emplace_back(v.Position);
        triangles.Faces.reserve(mesh.FaceCount());
        for (const auto fh : mesh.faces()) {
            std::vector<uint32_t> face_indices;
            for (const auto vh : mesh.fv_range(fh)) face_indices.emplace_back(*vh);
            triangles.Faces.emplace_back(std::move(face_indices));
        }

        // Populate a TriangleAttrs slot iff any source primitive had that bit set. Per-primitive
        // emission re-checks via `MeshPrimitives::AttributeFlags` at save time.
        uint32_t any_flags = 0;
        for (const auto f : layout.AttributeFlags) any_flags |= f;
        ::MeshVertexAttributes attrs;
        attrs.Colors0ComponentCount = layout.Colors0ComponentCount;
        const auto fill = [&]<typename V>(uint32_t bit, std::optional<std::vector<V>> &dest, V Vertex::*field) {
            if (!(any_flags & bit)) return;
            dest.emplace();
            dest->reserve(vertex_count);
            for (const auto &v : vertices) dest->emplace_back(v.*field);
        };
        fill(MeshAttributeBit_Normal, attrs.Normals, &Vertex::Normal);
        fill(MeshAttributeBit_Tangent, attrs.Tangents, &Vertex::Tangent);
        fill(MeshAttributeBit_Color0, attrs.Colors0, &Vertex::Color);
        fill(MeshAttributeBit_TexCoord0, attrs.TexCoords0, &Vertex::TexCoord0);
        fill(MeshAttributeBit_TexCoord1, attrs.TexCoords1, &Vertex::TexCoord1);
        fill(MeshAttributeBit_TexCoord2, attrs.TexCoords2, &Vertex::TexCoord2);
        fill(MeshAttributeBit_TexCoord3, attrs.TexCoords3, &Vertex::TexCoord3);

        const auto face_primitives = meshes.GetFacePrimitiveIndices(store_id);
        const auto primitive_materials = meshes.GetPrimitiveMaterialIndices(store_id);
        // Reverse populate's +1 material-index shift; `~0u` (out-of-range) = don't emit.
        ::MeshPrimitives prims{
            .FacePrimitiveIndices = {face_primitives.begin(), face_primitives.end()},
            .VertexCounts = layout.VertexCounts,
            .AttributeFlags = layout.AttributeFlags,
            .HasSourceIndices = layout.HasSourceIndices,
            .VariantMappings = layout.VariantMappings,
        };
        prims.MaterialIndices.reserve(primitive_materials.size());
        for (const auto reg_idx : primitive_materials) prims.MaterialIndices.emplace_back(reg_idx >= 1 ? reg_idx - 1 : ~0u);

        std::optional<ArmatureDeformData> deform_data;
        if (const auto bd_range = meshes.GetBoneDeformRange(store_id); bd_range.Count > 0) {
            const auto bd_span = meshes.BoneDeformBuffer.Get(bd_range);
            ArmatureDeformData dd;
            dd.Joints.reserve(bd_span.size());
            dd.Weights.reserve(bd_span.size());
            for (const auto &bdv : bd_span) {
                dd.Joints.emplace_back(bdv.Joints);
                dd.Weights.emplace_back(bdv.Weights);
            }
            deform_data = std::move(dd);
        }

        std::optional<MorphTargetData> morph_data;
        if (const auto target_count = meshes.GetMorphTargetCount(store_id); target_count > 0 && vertex_count > 0) {
            const auto mt_span = meshes.MorphTargetBuffer.Get(meshes.GetMorphTargetRange(store_id));
            MorphTargetData md{.TargetCount = target_count};
            md.PositionDeltas.reserve(mt_span.size());
            for (const auto &m : mt_span) md.PositionDeltas.emplace_back(m.PositionDelta);
            // CreateMesh writes 0 when source lacked normal deltas, so any non-zero ⇒ source had them.
            if (std::ranges::any_of(mt_span, [](const auto &m) { return m.NormalDelta != vec3{0}; })) {
                md.NormalDeltas.reserve(mt_span.size());
                for (const auto &m : mt_span) md.NormalDeltas.emplace_back(m.NormalDelta);
            }
            md.TangentDeltas = layout.MorphTangentDeltas;
            const auto default_weights = meshes.GetDefaultMorphWeights(store_id);
            md.DefaultWeights.assign(default_weights.begin(), default_weights.end());
            morph_data = std::move(md);
        }

        const auto *mn = r.try_get<const MeshName>(entity);
        scene.Meshes[smi.Value] = gltf::MeshData{
            .Triangles = std::move(triangles),
            .TriangleAttrs = std::move(attrs),
            .TrianglePrimitives = std::move(prims),
            .DeformData = std::move(deform_data),
            .MorphData = std::move(morph_data),
            .Name = mn ? mn->Value : std::string{},
        };
    }

    // Lines / Points entities → fill `.Lines` / `.Points` on the corresponding scene.Meshes slot.
    // Lines/Points entities → fill the matching `.Lines`/`.Points` slot. Attrs are recovered by
    // detecting non-default values: NORMAL non-zero, COLOR_0 not (1,1,1,1).
    for (const auto [entity, smi, kind] : r.view<const SourceMeshIndex, const SourceMeshKind>().each()) {
        if (kind.Value == MeshKind::Triangles) continue;
        const auto *mesh_ptr = r.try_get<const Mesh>(entity);
        if (!mesh_ptr) continue;
        const auto &mesh = *mesh_ptr;
        const auto vertices = meshes.GetVertices(mesh.GetStoreId());
        ::MeshData md;
        md.Positions.reserve(vertices.size());
        for (const auto &v : vertices) md.Positions.emplace_back(v.Position);
        if (kind.Value == MeshKind::Lines) {
            md.Edges.reserve(mesh.EdgeCount());
            for (const auto eh : mesh.edges()) {
                const auto h0 = mesh.GetHalfedge(eh, 0);
                md.Edges.emplace_back(std::array<uint32_t, 2>{*mesh.GetFromVertex(h0), *mesh.GetToVertex(h0)});
            }
        }
        ::MeshVertexAttributes attrs;
        const auto fill_if_any = [&]<typename V>(std::optional<std::vector<V>> &dest, V Vertex::*field, V sentinel) {
            if (!std::ranges::any_of(vertices, [&](const auto &v) { return v.*field != sentinel; })) return;
            dest.emplace();
            dest->reserve(vertices.size());
            for (const auto &v : vertices) dest->emplace_back(v.*field);
        };
        fill_if_any(attrs.Normals, &Vertex::Normal, vec3{0});
        fill_if_any(attrs.Colors0, &Vertex::Color, vec4{1});
        if (attrs.Colors0) attrs.Colors0ComponentCount = 4; // CPU stores vec4 regardless of source
        auto &dst = scene.Meshes[smi.Value];
        if (kind.Value == MeshKind::Lines) {
            dst.Lines = std::move(md);
            dst.LineAttrs = std::move(attrs);
        } else {
            dst.Points = std::move(md);
            dst.PointAttrs = std::move(attrs);
        }
        if (dst.Name.empty()) {
            if (const auto *mn = r.try_get<const MeshName>(entity)) dst.Name = mn->Value;
        }
    }

    // Cameras / lights: one entry per component-bearing entity, in source-aligned order. Per the
    // Khronos sample set source cameras/lights aren't shared across nodes, so 1:1 matches source counts.
    std::unordered_map<entt::entity, uint32_t> camera_entity_to_index, light_entity_to_index;
    {
        auto camera_view = r.view<const ::Camera>();
        for (const auto &[_, entity] : ordered_by_source.operator()<SourceCameraIndex>(camera_view)) {
            camera_entity_to_index[entity] = uint32_t(scene.Cameras.size());
            const auto *cn = r.try_get<const CameraName>(entity);
            scene.Cameras.emplace_back(gltf::Camera{.Camera = camera_view.get<const ::Camera>(entity), .Name = cn ? cn->Value : std::string{}});
        }
        auto light_view = r.view<const PunctualLight>();
        for (const auto &[_, entity] : ordered_by_source.operator()<SourceLightIndex>(light_view)) {
            light_entity_to_index[entity] = uint32_t(scene.Lights.size());
            const auto *ln = r.try_get<const LightName>(entity);
            scene.Lights.emplace_back(gltf::Light{.Light = light_view.get<const PunctualLight>(entity), .Name = ln ? ln->Value : std::string{}});
        }
    }

    // Armature data entities → SkinIndex (source-fidelity from ImportedSkin if present).
    std::unordered_map<entt::entity, uint32_t> armature_data_to_skin_index;
    for (const auto e : r.view<const Armature>()) {
        const auto &arm = r.get<const Armature>(e);
        armature_data_to_skin_index[e] = arm.ImportedSkin ? arm.ImportedSkin->SkinIndex : uint32_t(armature_data_to_skin_index.size());
    }

    // Map data entity → ArmatureObject entity (for parent-of-armature lookup + name).
    std::unordered_map<entt::entity, entt::entity> armature_data_to_object;
    for (const auto [e, ao] : r.view<const ArmatureObject>().each()) armature_data_to_object[ao.Entity] = e;

    // Skins: one gltf::Skin per Armature data entity carrying ImportedSkin metadata.
    for (const auto data_entity : r.view<const Armature>()) {
        const auto &arm = r.get<const Armature>(data_entity);
        if (!arm.ImportedSkin) continue;
        const auto skin_index = arm.ImportedSkin->SkinIndex;
        if (skin_index >= scene.Skins.size()) scene.Skins.resize(skin_index + 1);
        auto &gltf_skin = scene.Skins[skin_index];
        gltf_skin.SkinIndex = skin_index;
        gltf_skin.SkeletonNodeIndex = arm.ImportedSkin->SkeletonNodeIndex;
        gltf_skin.AnchorNodeIndex = arm.ImportedSkin->AnchorNodeIndex;
        gltf_skin.InverseBindMatrices = arm.ImportedSkin->InverseBindMatrices;
        if (const auto ait = armature_data_to_object.find(data_entity); ait != armature_data_to_object.end()) {
            if (const auto *sn = r.try_get<const SceneNode>(ait->second); sn && sn->Parent != null_entity) {
                if (const auto *psni = r.try_get<const SourceNodeIndex>(sn->Parent)) gltf_skin.ParentObjectNodeIndex = psni->Value;
            }
            if (const auto *sn = r.try_get<const SkinName>(ait->second)) gltf_skin.Name = sn->Value;
            else if (!r.all_of<SourceEmptyName>(ait->second)) {
                if (const auto *name = r.try_get<const Name>(ait->second)) gltf_skin.Name = name->Value;
            }
        }
        gltf_skin.Joints.reserve(arm.ImportedSkin->OrderedJointNodeIndices.size());
        for (const auto joint_node_index : arm.ImportedSkin->OrderedJointNodeIndices) {
            std::optional<uint32_t> parent_joint_node_index;
            std::string joint_name;
            Transform rest_local{};
            for (const auto &bone : arm.Bones) {
                if (bone.JointNodeIndex == joint_node_index) {
                    joint_name = bone.Name;
                    rest_local = bone.RestLocal;
                    if (bone.ParentIndex != InvalidBoneIndex) parent_joint_node_index = arm.Bones[bone.ParentIndex].JointNodeIndex;
                    break;
                }
            }
            gltf_skin.Joints.emplace_back(gltf::SkinJoint{
                .JointNodeIndex = joint_node_index,
                .ParentJointNodeIndex = parent_joint_node_index,
                .RestLocal = rest_local,
                .Name = std::move(joint_name),
            });
        }
    }

    // Scene tree: only source-derived entities (with `SourceNodeIndex`) become nodes; engine
    // helpers like the armature object are skipped. Hierarchy comes from `SourceParentNodeIndex`,
    // not `SceneNode` (which gets mutated by skinning/armature re-parenting at populate).
    std::unordered_map<entt::entity, uint32_t> entity_to_node_index;
    uint32_t total_node_count = 0;
    for (const auto [e, sni] : r.view<const SourceNodeIndex>().each()) {
        entity_to_node_index[e] = sni.Value;
        total_node_count = std::max(total_node_count, sni.Value + 1u);
    }
    // Children paired with sibling position so we sort in source order.
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> children_by_parent;
    for (const auto [e, sni, spi] : r.view<const SourceNodeIndex, const SourceParentNodeIndex>().each()) {
        const auto *ssi = r.try_get<const SourceSiblingIndex>(e);
        children_by_parent[spi.Value].emplace_back(ssi ? ssi->Value : sni.Value, sni.Value);
    }
    for (auto &[_, kids] : children_by_parent) std::ranges::sort(kids, {}, &std::pair<uint32_t, uint32_t>::first);

    scene.Nodes.resize(total_node_count);
    for (const auto [entity, node_index] : entity_to_node_index) {
        auto &node = scene.Nodes[node_index];
        node.NodeIndex = node_index;
        // Prefer the SourceLocalTransform sidecar when present (lossless). Fallback: decompose
        // `inv(parent_world) * Transform` from the engine's world-space `Transform`.
        if (const auto *slt = r.try_get<const SourceLocalTransform>(entity)) {
            node.LocalTransform = slt->Value;
        } else {
            const auto &transform = r.get<const Transform>(entity);
            const auto *pi = r.try_get<const ParentInverse>(entity);
            node.LocalTransform = pi ? ToTransform(pi->M * ToMatrix(transform)) : transform;
        }
        node.WorldTransform = r.get<const WorldTransform>(entity);
        node.InScene = true;
        node.IsJoint = r.all_of<BoneIndex>(entity);
        if (const auto *smt = r.try_get<const SourceMatrixTransform>(entity)) node.SourceMatrix = smt->Value;
        if (!r.all_of<SourceEmptyName>(entity)) {
            if (const auto *son = r.try_get<const SourceObjectName>(entity)) node.Name = son->Value;
            else if (const auto *name = r.try_get<const Name>(entity)) node.Name = name->Value;
        }
        if (const auto *spi = r.try_get<const SourceParentNodeIndex>(entity)) node.ParentNodeIndex = spi->Value;
        if (const auto it = children_by_parent.find(node_index); it != children_by_parent.end()) {
            node.ChildrenNodeIndices.reserve(it->second.size());
            for (const auto &[_, child_idx] : it->second) node.ChildrenNodeIndices.push_back(child_idx);
        }
    }

    // Fill mesh/camera/light/skin refs on both nodes and objects from the entity-index maps.
    // `ArmatureModifier::ArmatureEntity` actually holds the Armature *data* entity (legacy naming).
    const auto fill_refs = [&](entt::entity entity, auto &dst) {
        if (const auto *inst = r.try_get<const Instance>(entity)) {
            if (const auto it = mesh_entity_to_index.find(inst->Entity); it != mesh_entity_to_index.end()) dst.MeshIndex = it->second;
        }
        if (const auto it = camera_entity_to_index.find(entity); it != camera_entity_to_index.end()) dst.CameraIndex = it->second;
        if (const auto it = light_entity_to_index.find(entity); it != light_entity_to_index.end()) dst.LightIndex = it->second;
        if (const auto *am = r.try_get<const ArmatureModifier>(entity)) {
            if (const auto it = armature_data_to_skin_index.find(am->ArmatureEntity); it != armature_data_to_skin_index.end()) dst.SkinIndex = it->second;
        }
    };
    for (const auto [entity, node_index] : entity_to_node_index) fill_refs(entity, scene.Nodes[node_index]);

    const auto to_object_type = [](ObjectType k) {
        switch (k) {
            case ObjectType::Mesh: return gltf::Object::Type::Mesh;
            case ObjectType::Camera: return gltf::Object::Type::Camera;
            case ObjectType::Light: return gltf::Object::Type::Light;
            default: return gltf::Object::Type::Empty;
        }
    };
    auto object_view = r.view<const Transform, const ObjectKind>();
    for (const auto entity : object_view) {
        const auto kind = object_view.get<const ObjectKind>(entity).Value;
        if (kind == ObjectType::Armature) continue; // → gltf::Skin, handled separately.
        const auto it = entity_to_node_index.find(entity);
        if (it == entity_to_node_index.end()) continue;
        const auto *spi = r.try_get<const SourceParentNodeIndex>(entity);
        gltf::Object obj{
            .ObjectType = to_object_type(kind),
            .NodeIndex = it->second,
            .ParentNodeIndex = spi ? std::optional{spi->Value} : std::nullopt,
            .WorldTransform = r.get<const WorldTransform>(entity),
        };
        if (const auto *name = r.try_get<const Name>(entity)) obj.Name = name->Value;
        fill_refs(entity, obj);
        scene.Objects.emplace_back(std::move(obj));
    }

    // Physics document-level resources, source-aligned via the per-resource index sidecars.
    std::unordered_map<entt::entity, uint32_t> physics_material_to_index, physics_jointdef_to_index, collision_filter_to_index;
    {
        auto mat_view = r.view<const PhysicsMaterial>();
        for (const auto &[_, e] : ordered_by_source.operator()<SourcePhysicsMaterialIndex>(mat_view)) {
            physics_material_to_index[e] = uint32_t(scene.PhysicsMaterials.size());
            scene.PhysicsMaterials.emplace_back(mat_view.get<const PhysicsMaterial>(e));
        }
        auto jd_view = r.view<const ::PhysicsJointDef>();
        for (const auto &[_, e] : ordered_by_source.operator()<SourcePhysicsJointDefIndex>(jd_view)) {
            physics_jointdef_to_index[e] = uint32_t(scene.PhysicsJointDefs.size());
            scene.PhysicsJointDefs.emplace_back(jd_view.get<const ::PhysicsJointDef>(e));
        }
        const auto resolve_system_names = [&](std::span<const entt::entity> systems) {
            std::vector<std::string> names;
            names.reserve(systems.size());
            for (const auto se : systems) {
                if (const auto *cs = r.try_get<const CollisionSystem>(se)) names.emplace_back(cs->Name);
            }
            return names;
        };
        auto cf_view = r.view<const CollisionFilter>();
        for (const auto &[_, e] : ordered_by_source.operator()<SourceCollisionFilterIndex>(cf_view)) {
            const auto &f = cf_view.get<const CollisionFilter>(e);
            collision_filter_to_index[e] = uint32_t(scene.CollisionFilters.size());
            gltf::CollisionFilterData data{.CollisionSystems = resolve_system_names(f.Systems), .Name = f.Name};
            if (f.Mode == CollideMode::Allowlist) data.CollideWithSystems = resolve_system_names(f.CollideSystems);
            else if (f.Mode == CollideMode::Blocklist) data.NotCollideWithSystems = resolve_system_names(f.CollideSystems);
            scene.CollisionFilters.emplace_back(std::move(data));
        }
    }

    // Per-node physics: walk all nodes, fill Motion/Velocity/Collider/Material/Trigger/Joint.
    for (const auto [entity, node_index] : entity_to_node_index) {
        if (node_index >= scene.Nodes.size()) continue;
        auto &node = scene.Nodes[node_index];
        if (const auto *m = r.try_get<const PhysicsMotion>(entity)) node.Motion = *m;
        if (const auto *v = r.try_get<const PhysicsVelocity>(entity)) node.Velocity = *v;
        if (const auto *cs = r.try_get<const ColliderShape>(entity)) {
            if (!r.all_of<TriggerTag>(entity)) {
                node.Collider = *cs;
                if (const auto *cm = r.try_get<const ColliderMaterial>(entity)) {
                    gltf::Node::MaterialRefs refs;
                    if (const auto mit = physics_material_to_index.find(cm->PhysicsMaterialEntity); mit != physics_material_to_index.end()) refs.PhysicsMaterialIndex = mit->second;
                    if (const auto fit = collision_filter_to_index.find(cm->CollisionFilterEntity); fit != collision_filter_to_index.end()) refs.CollisionFilterIndex = fit->second;
                    if (refs.PhysicsMaterialIndex || refs.CollisionFilterIndex) node.Material = refs;
                }
                if (cs->MeshEntity != null_entity) {
                    if (const auto mit = mesh_entity_to_index.find(cs->MeshEntity); mit != mesh_entity_to_index.end()) node.ColliderGeometryMeshIndex = mit->second;
                }
            } else {
                // Geometry trigger: shape on the same entity, distinguished by TriggerTag.
                gltf::Node::TriggerData td{.Shape = cs->Shape};
                if (cs->MeshEntity != null_entity) {
                    if (const auto mit = mesh_entity_to_index.find(cs->MeshEntity); mit != mesh_entity_to_index.end()) td.GeometryMeshIndex = mit->second;
                }
                if (const auto *cm = r.try_get<const ColliderMaterial>(entity)) {
                    if (const auto fit = collision_filter_to_index.find(cm->CollisionFilterEntity); fit != collision_filter_to_index.end()) td.CollisionFilterIndex = fit->second;
                }
                node.Trigger = std::move(td);
            }
        }
        if (const auto *tn = r.try_get<const TriggerNodes>(entity)) {
            gltf::Node::TriggerData td;
            td.NodeIndices.reserve(tn->Nodes.size());
            for (const auto ne : tn->Nodes) {
                if (const auto nit = entity_to_node_index.find(ne); nit != entity_to_node_index.end()) td.NodeIndices.emplace_back(nit->second);
            }
            if (const auto fit = collision_filter_to_index.find(tn->CollisionFilterEntity); fit != collision_filter_to_index.end()) td.CollisionFilterIndex = fit->second;
            node.Trigger = std::move(td);
        }
        if (const auto *pj = r.try_get<const PhysicsJoint>(entity)) {
            gltf::Node::JointData jd;
            if (const auto cit = entity_to_node_index.find(pj->ConnectedNode); cit != entity_to_node_index.end()) jd.ConnectedNodeIndex = cit->second;
            if (const auto dit = physics_jointdef_to_index.find(pj->JointDefEntity); dit != physics_jointdef_to_index.end()) jd.JointDefIndex = dit->second;
            jd.EnableCollision = pj->EnableCollision;
            node.Joint = jd;
        }
    }

    // Animations: merge per-entity engine clips back into source-side clips by (Name, Duration).
    // Populate split each source clip into N engine clips (one per affected entity); reverse here.
    // Pre-seed scene.Animations in source-name order from `GltfSourceAssets::AnimationOrder` so
    // animations[*] indices match source; clips not present in the source list (runtime-added)
    // are appended.
    std::unordered_map<std::string, size_t> clip_index_by_name;
    if (src_assets) {
        scene.Animations.reserve(src_assets->AnimationOrder.size());
        for (const auto &name : src_assets->AnimationOrder) {
            clip_index_by_name.emplace(name, scene.Animations.size());
            scene.Animations.emplace_back(gltf::AnimationClip{.Name = name, .DurationSeconds = 0.f, .Channels = {}});
        }
    }
    const auto get_or_create_clip = [&](const std::string &name, float duration) -> gltf::AnimationClip & {
        auto [it, inserted] = clip_index_by_name.try_emplace(name, scene.Animations.size());
        if (inserted) scene.Animations.emplace_back(gltf::AnimationClip{.Name = name, .DurationSeconds = duration, .Channels = {}});
        else scene.Animations[it->second].DurationSeconds = std::max(scene.Animations[it->second].DurationSeconds, duration);
        return scene.Animations[it->second];
    };
    const auto get_node_index = [&](entt::entity e) -> std::optional<uint32_t> {
        const auto it = entity_to_node_index.find(e);
        return it != entity_to_node_index.end() ? std::optional<uint32_t>{it->second} : std::nullopt;
    };

    // Armature animation: bone channels → joint node index.
    for (const auto [data_entity, anim] : r.view<const ArmatureAnimation>().each()) {
        const auto &arm = r.get<const Armature>(data_entity);
        for (const auto &clip : anim.Clips) {
            auto &gclip = get_or_create_clip(clip.Name, clip.DurationSeconds);
            for (const auto &ch : clip.Channels) {
                if (ch.BoneIndex == InvalidBoneIndex || ch.BoneIndex >= arm.Bones.size()) continue;
                const auto &bone = arm.Bones[ch.BoneIndex];
                if (!bone.JointNodeIndex) continue;
                gclip.Channels.emplace_back(gltf::AnimationChannel{
                    .TargetNodeIndex = *bone.JointNodeIndex,
                    .Target = ch.Target,
                    .Interp = ch.Interp,
                    .TimesSeconds = ch.TimesSeconds,
                    .Values = ch.Values,
                });
            }
        }
    }
    // Morph weight animation: target = the mesh-instance entity's node index.
    for (const auto [entity, anim] : r.view<const MorphWeightAnimation>().each()) {
        const auto node_idx = get_node_index(entity);
        if (!node_idx) continue;
        for (const auto &clip : anim.Clips) {
            auto &gclip = get_or_create_clip(clip.Name, clip.DurationSeconds);
            for (const auto &ch : clip.Channels) {
                gclip.Channels.emplace_back(gltf::AnimationChannel{
                    .TargetNodeIndex = *node_idx,
                    .Target = AnimationPath::Weights,
                    .Interp = ch.Interp,
                    .TimesSeconds = ch.TimesSeconds,
                    .Values = ch.Values,
                });
            }
        }
    }
    // Node transform animation: target = the object entity's node index.
    for (const auto [entity, anim] : r.view<const NodeTransformAnimation>().each()) {
        const auto node_idx = get_node_index(entity);
        if (!node_idx) continue;
        for (const auto &clip : anim.Clips) {
            auto &gclip = get_or_create_clip(clip.Name, clip.DurationSeconds);
            for (const auto &ch : clip.Channels) {
                gclip.Channels.emplace_back(gltf::AnimationChannel{
                    .TargetNodeIndex = *node_idx,
                    .Target = ch.Target,
                    .Interp = ch.Interp,
                    .TimesSeconds = ch.TimesSeconds,
                    .Values = ch.Values,
                });
            }
        }
    }

    return scene;
}
