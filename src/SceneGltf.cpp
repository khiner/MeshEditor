#include "MeshInstance.h"
#include "Scene.h"
#include "SceneMaterials.h"
#include "SceneTextures.h"
#include "SceneTree.h"
#include "gltf/GltfLoader.h"
#include "mesh/MeshStore.h"
#include "mesh/MorphTargetData.h"

#include <entt/entity/registry.hpp>
#include <iostream>

std::expected<std::pair<entt::entity, entt::entity>, std::string> Scene::AddGltfScene(const std::filesystem::path &path) {
    auto scene = gltf::LoadScene(path);
    if (!scene) return std::unexpected{std::move(scene.error())};

    auto &texture_store = *Textures;
    const auto texture_start = texture_store.Textures.size();
    const auto material_start = GetMaterialCount(*Buffers);
    const auto material_name_start = R.get<const MaterialStore>(SceneEntity).Names.size();
    const auto rollback_import_side_effects = [&] {
        if (texture_store.Textures.size() > texture_start) {
            ReleaseSamplerSlots(*Slots, CollectSamplerSlots(std::span<const TextureEntry>{texture_store.Textures}.subspan(texture_start)));
            texture_store.Textures.resize(texture_start);
        }
        if (GetMaterialCount(*Buffers) > material_start) SetMaterialCount(*Buffers, material_start);
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

    std::unordered_map<uint64_t, uint32_t> texture_slot_cache;
    const auto texture_cache_key = [](uint32_t texture_index, TextureColorSpace color_space) {
        return (uint64_t(texture_index) << 1u) | (color_space == TextureColorSpace::Srgb ? 1u : 0u);
    };
    const auto resolve_texture_slot = [&](uint32_t texture_index, TextureColorSpace color_space) -> std::expected<uint32_t, std::string> {
        if (texture_index >= scene->Textures.size()) return InvalidSlot;
        const auto cache_key = texture_cache_key(texture_index, color_space);
        if (const auto it = texture_slot_cache.find(cache_key); it != texture_slot_cache.end()) return it->second;

        const auto &src_texture = scene->Textures[texture_index];
        const auto image_index = resolve_image_index(src_texture);
        if (!image_index || *image_index >= scene->Images.size()) return InvalidSlot;

        const auto &src_image = scene->Images[*image_index];
        const auto *src_sampler = src_texture.SamplerIndex && *src_texture.SamplerIndex < scene->Samplers.size() ?
            &scene->Samplers[*src_texture.SamplerIndex] :
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

        auto texture = CreateTextureEntryFromEncoded(
            Vk, GetBufferCtx(Buffers.get()), *CommandPool, *OneShotFence, *Slots,
            src_image.Bytes,
            src_image.Name.empty() ? std::format("Image{}", *image_index) : src_image.Name,
            texture_name,
            color_space,
            wrap_s, wrap_t,
            sampler_config
        );
        if (!texture) return std::unexpected{std::move(texture.error())};
        const auto sampler_slot = texture->SamplerSlot;
        texture_store.Textures.emplace_back(std::move(*texture));
        texture_slot_cache.emplace(cache_key, sampler_slot);
        return sampler_slot;
    };

    std::vector<uint32_t> material_indices_by_gltf_material(scene->Materials.size(), 0u);
    const auto material_count = GetMaterialCount(*Buffers);
    const auto default_material_index = material_count > 0 ? material_count - 1u : 0u;
    std::vector<std::string> material_names;
    material_names.reserve(scene->Materials.size());
    for (uint32_t material_index = 0; material_index < scene->Materials.size(); ++material_index) {
        const auto &src_named_material = scene->Materials[material_index];
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
        PBRMaterial gpu_material = src_material;
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
        material_indices_by_gltf_material[material_index] = AppendMaterial(*Buffers, gpu_material);
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

    std::vector<entt::entity> mesh_entities;
    mesh_entities.reserve(scene->Meshes.size());
    // Track morph data per mesh for later component setup
    std::vector<std::optional<MorphTargetData>> mesh_morphs;
    mesh_morphs.reserve(scene->Meshes.size());
    entt::entity first_mesh_entity = entt::null;
    // Per-mesh: optional line entity + optional point entity
    struct ExtraPrimitiveEntities {
        entt::entity Lines{entt::null}, Points{entt::null};
    };
    std::vector<ExtraPrimitiveEntities> extra_entities_per_mesh(scene->Meshes.size());
    for (uint32_t mi = 0; mi < scene->Meshes.size(); ++mi) {
        auto &scene_mesh = scene->Meshes[mi];
        entt::entity mesh_entity = entt::null;
        if (scene_mesh.Triangles) {
            if (scene_mesh.Triangles->PrimitiveMaterialIndices) {
                for (auto &local_material_index : *scene_mesh.Triangles->PrimitiveMaterialIndices) {
                    local_material_index = local_material_index < material_indices_by_gltf_material.size() ?
                        material_indices_by_gltf_material[local_material_index] :
                        fallback_material_index;
                }
            }
            auto morph_data_copy = scene_mesh.MorphData; // Keep a copy for component setup
            auto mesh = Meshes->CreateMesh(std::move(*scene_mesh.Triangles), std::move(scene_mesh.DeformData), std::move(scene_mesh.MorphData));
            const auto [me, _] = AddMesh(std::move(mesh), std::nullopt);
            mesh_entity = me;
            R.emplace<Path>(mesh_entity, path);
            mesh_morphs.emplace_back(std::move(morph_data_copy));
        } else {
            mesh_morphs.emplace_back(std::nullopt);
        }
        if (first_mesh_entity == entt::null && mesh_entity != entt::null) first_mesh_entity = mesh_entity;
        mesh_entities.emplace_back(mesh_entity);

        auto create_extra = [&](std::optional<MeshData> &data) -> entt::entity {
            if (!data) return entt::null;
            auto m = Meshes->CreateMesh(std::move(*data));
            const auto [e, _] = AddMesh(std::move(m), std::nullopt);
            R.emplace<Path>(e, path);
            if (first_mesh_entity == entt::null) first_mesh_entity = e;
            return e;
        };
        auto lines_entity = create_extra(scene_mesh.Lines);
        auto points_entity = create_extra(scene_mesh.Points);
        extra_entities_per_mesh[mi] = {lines_entity, points_entity};
    }

    const auto name_prefix = path.stem().string();
    std::unordered_map<uint32_t, entt::entity> object_entities_by_node;
    object_entities_by_node.reserve(scene->Objects.size());
    std::unordered_map<uint32_t, std::vector<entt::entity>> skinned_mesh_instances_by_skin;
    skinned_mesh_instances_by_skin.reserve(scene->Skins.size());

    entt::entity first_object_entity = entt::null,
                 first_mesh_object_entity = entt::null,
                 first_root_empty_entity = entt::null,
                 first_armature_entity = entt::null;
    for (uint32_t i = 0; i < scene->Objects.size(); ++i) {
        const auto &object = scene->Objects[i];
        const auto object_name = object.Name.empty() ? std::format("{}_{}", name_prefix, i) : object.Name;
        entt::entity object_entity = entt::null;
        if (object.ObjectType == gltf::Object::Type::Mesh &&
            object.MeshIndex &&
            *object.MeshIndex < mesh_entities.size() &&
            mesh_entities[*object.MeshIndex] != entt::null) {
            object_entity = AddMeshInstance(
                mesh_entities[*object.MeshIndex],
                {.Name = object_name, .Transform = object.WorldTransform, .Select = MeshInstanceCreateInfo::SelectBehavior::None, .Visible = true}
            );
        } else if (object.ObjectType == gltf::Object::Type::Camera && object.CameraIndex && *object.CameraIndex < scene->Cameras.size()) {
            object_entity = AddCamera({.Name = object_name, .Transform = object.WorldTransform, .Select = MeshInstanceCreateInfo::SelectBehavior::None});
            const auto &scd = scene->Cameras[*object.CameraIndex];
            R.replace<Camera>(object_entity, scd.Camera);
        } else if (object.ObjectType == gltf::Object::Type::Light && object.LightIndex && *object.LightIndex < scene->Lights.size()) {
            const auto &sld = scene->Lights[*object.LightIndex];
            object_entity = AddLight({.Name = object_name, .Transform = object.WorldTransform, .Select = MeshInstanceCreateInfo::SelectBehavior::None}, sld.Light);
        } else {
            object_entity = AddEmpty({.Name = object_name, .Transform = object.WorldTransform, .Select = MeshInstanceCreateInfo::SelectBehavior::None});
        }
        // Create instances for non-triangle primitives (lines/points) associated with this mesh
        if (object.ObjectType == gltf::Object::Type::Mesh && object.MeshIndex && *object.MeshIndex < extra_entities_per_mesh.size()) {
            const auto &extras = extra_entities_per_mesh[*object.MeshIndex];
            for (const auto extra_entity : {extras.Lines, extras.Points}) {
                if (extra_entity == entt::null) continue;
                AddMeshInstance(extra_entity, {.Name = object_name, .Transform = object.WorldTransform, .Select = MeshInstanceCreateInfo::SelectBehavior::None, .Visible = true});
            }
        }

        object_entities_by_node[object.NodeIndex] = object_entity;
        // glTF node.skin is deform linkage, not a transform-parent relationship.
        if (object.SkinIndex && R.all_of<MeshInstance>(object_entity)) skinned_mesh_instances_by_skin[*object.SkinIndex].emplace_back(object_entity);
        if (first_object_entity == entt::null) first_object_entity = object_entity;
        if (first_mesh_object_entity == entt::null && object.ObjectType == gltf::Object::Type::Mesh) first_mesh_object_entity = object_entity;
        if (first_root_empty_entity == entt::null && object.ObjectType == gltf::Object::Type::Empty && !object.ParentNodeIndex) first_root_empty_entity = object_entity;
    }

    for (const auto &object : scene->Objects) {
        if (!object.ParentNodeIndex) continue;

        const auto child_it = object_entities_by_node.find(object.NodeIndex);
        if (child_it == object_entities_by_node.end()) continue;
        const auto parent_it = object_entities_by_node.find(*object.ParentNodeIndex);
        if (parent_it == object_entities_by_node.end()) continue;
        SetParent(R, child_it->second, parent_it->second);
    }

    std::unordered_map<uint32_t, const gltf::Node *> scene_nodes_by_index;
    scene_nodes_by_index.reserve(scene->Nodes.size());
    for (const auto &node : scene->Nodes) scene_nodes_by_index.emplace(node.NodeIndex, &node);

    for (const auto &skin : scene->Skins) {
        const auto armature_data_entity = R.create();
        auto &armature = R.emplace<Armature>(armature_data_entity);

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
                R.all_of<MeshInstance>(object_it->second) &&
                !R.all_of<BoneAttachment>(object_it->second)) {
                R.emplace<BoneAttachment>(object_it->second, armature_data_entity, bone_id);
            }
            imported_skin.OrderedJointNodeIndices.emplace_back(joint.JointNodeIndex);
        }
        armature.FinalizeImportedStructure();

        imported_skin.InverseBindMatrices.resize(imported_skin.OrderedJointNodeIndices.size(), I4);
        armature.ImportedSkin = std::move(imported_skin);
        if (!skin.AnchorNodeIndex) {
            return std::unexpected{std::format("glTF import failed for '{}': skin {} has no deterministic anchor node.", path.string(), skin.SkinIndex)};
        }

        const auto anchor_it = scene_nodes_by_index.find(*skin.AnchorNodeIndex);
        if (anchor_it == scene_nodes_by_index.end() || !anchor_it->second->InScene) {
            return std::unexpected{std::format("glTF import failed for '{}': skin {} anchor node {} is not in the imported scene.", path.string(), skin.SkinIndex, *skin.AnchorNodeIndex)};
        }

        const auto armature_entity = R.create();
        R.emplace<ObjectKind>(armature_entity, ObjectType::Armature);
        R.emplace<ArmatureObject>(armature_entity, armature_data_entity);
        const auto &t = anchor_it->second->WorldTransform;
        R.emplace<Position>(armature_entity, t.P);
        R.emplace<Rotation>(armature_entity, t.R);
        R.emplace<Scale>(armature_entity, t.S);
        UpdateWorldTransform(R, armature_entity);
        R.emplace<Name>(armature_entity, CreateName(R, skin.Name.empty() ? std::format("{}_Armature{}", name_prefix, skin.SkinIndex) : skin.Name));

        if (skin.ParentObjectNodeIndex) {
            if (const auto parent_it = object_entities_by_node.find(*skin.ParentObjectNodeIndex);
                parent_it != object_entities_by_node.end()) {
                SetParent(R, armature_entity, parent_it->second);
            }
        }

        if (first_armature_entity == entt::null) first_armature_entity = armature_entity;
        if (first_object_entity == entt::null) first_object_entity = armature_entity;

        if (const auto skinned_it = skinned_mesh_instances_by_skin.find(skin.SkinIndex);
            skinned_it != skinned_mesh_instances_by_skin.end()) {
            for (const auto mesh_instance_entity : skinned_it->second) {
                if (!R.valid(mesh_instance_entity) || !R.all_of<MeshInstance>(mesh_instance_entity)) continue;
                R.emplace_or_replace<ArmatureModifier>(mesh_instance_entity, armature_data_entity, armature_entity);
            }
        } else {
            return std::unexpected{std::format("glTF import failed '{}': skin {} is used but no mesh instances were emitted for skin binding.", path.string(), skin.SkinIndex)};
        }

        // Allocate pose state and GPU deform buffer for this armature
        {
            ArmaturePoseState pose_state;
            pose_state.BonePoseLocal.resize(armature.Bones.size());
            for (uint32_t i = 0; i < armature.Bones.size(); ++i) pose_state.BonePoseLocal[i] = armature.Bones[i].RestLocal;
            pose_state.GpuDeformRange = AllocateArmatureDeform(Buffers.get(), armature.ImportedSkin->OrderedJointNodeIndices.size());

            // Compute initial rest-pose deform matrices
            auto gpu_span = GetArmatureDeformMutable(Buffers.get(), pose_state.GpuDeformRange);
            ComputeDeformMatrices(armature, pose_state.BonePoseLocal, armature.ImportedSkin->InverseBindMatrices, gpu_span);

            R.emplace<ArmaturePoseState>(armature_data_entity, std::move(pose_state));
        }
    }

    // Resolve animation data: map glTF animation channels to bone indices
    for (auto &anim_clip : scene->Animations) {
        for (const auto &skin : scene->Skins) {
            entt::entity target_data_entity = entt::null;
            for (const auto [entity, ad] : R.view<Armature>().each()) {
                if (ad.ImportedSkin && ad.ImportedSkin->SkinIndex == skin.SkinIndex) {
                    target_data_entity = entity;
                    break;
                }
            }
            if (target_data_entity == entt::null) continue;

            const auto &ad = R.get<const Armature>(target_data_entity);
            AnimationClip resolved_clip{.Name = std::move(anim_clip.Name), .DurationSeconds = anim_clip.DurationSeconds, .Channels = {}};
            for (auto &ch : anim_clip.Channels) {
                if (const auto bone_id = ad.FindBoneIdByJointNodeIndex(ch.TargetNodeIndex)) {
                    if (const auto bone_index = ad.FindBoneIndex(*bone_id)) {
                        resolved_clip.Channels.emplace_back(AnimationChannel{
                            .BoneIndex = *bone_index,
                            .Target = ch.Target,
                            .Interp = ch.Interp,
                            .TimesSeconds = std::move(ch.TimesSeconds),
                            .Values = std::move(ch.Values),
                        });
                    }
                }
            }
            if (!resolved_clip.Channels.empty()) {
                if (auto *existing = R.try_get<ArmatureAnimation>(target_data_entity)) {
                    existing->Clips.emplace_back(std::move(resolved_clip));
                } else {
                    R.emplace<ArmatureAnimation>(target_data_entity, ArmatureAnimation{.Clips = {std::move(resolved_clip)}});
                }
            }
        }
    }

    // Set up morph weight state for mesh instances with morph targets
    // Build a map: node_index -> mesh instance entity, for resolving weight animation channels
    std::unordered_map<uint32_t, entt::entity> morph_instance_by_node;
    for (const auto &object : scene->Objects) {
        if (object.ObjectType != gltf::Object::Type::Mesh || !object.MeshIndex) continue;
        if (*object.MeshIndex >= mesh_morphs.size() || !mesh_morphs[*object.MeshIndex]) continue;
        const auto obj_it = object_entities_by_node.find(object.NodeIndex);
        if (obj_it == object_entities_by_node.end()) continue;
        const auto instance_entity = obj_it->second;
        if (!R.all_of<MeshInstance>(instance_entity)) continue;

        const auto &morph = *mesh_morphs[*object.MeshIndex];
        if (morph.TargetCount == 0) continue;

        MorphWeightState state;
        if (object.NodeWeights) {
            // Per-node morph weight overrides (glTF node.weights) take priority over mesh.weights
            state.Weights.resize(morph.TargetCount, 0.f);
            std::copy_n(object.NodeWeights->begin(), std::min(uint32_t(object.NodeWeights->size()), morph.TargetCount), state.Weights.begin());
        } else {
            state.Weights = morph.DefaultWeights;
        }
        state.GpuWeightRange = AllocateMorphWeights(Buffers.get(), morph.TargetCount);
        auto gpu_weights = GetMorphWeightsMutable(Buffers.get(), state.GpuWeightRange);
        std::copy(state.Weights.begin(), state.Weights.end(), gpu_weights.begin());
        R.emplace<MorphWeightState>(instance_entity, std::move(state));
        morph_instance_by_node[object.NodeIndex] = instance_entity;
    }

    // Resolve morph weight animation channels
    for (auto &anim_clip : scene->Animations) {
        // Group weight channels by target node
        std::unordered_map<uint32_t, std::vector<gltf::AnimationChannel *>> weight_channels_by_node;
        for (auto &ch : anim_clip.Channels) {
            if (ch.Target == AnimationPath::Weights) weight_channels_by_node[ch.TargetNodeIndex].emplace_back(&ch);
        }
        for (auto &[node_index, channels] : weight_channels_by_node) {
            const auto inst_it = morph_instance_by_node.find(node_index);
            if (inst_it == morph_instance_by_node.end()) continue;
            const auto instance_entity = inst_it->second;

            MorphWeightClip resolved_clip{.Name = anim_clip.Name, .DurationSeconds = anim_clip.DurationSeconds, .Channels = {}};
            for (auto *ch : channels) {
                resolved_clip.Channels.emplace_back(MorphWeightChannel{
                    .Interp = ch->Interp,
                    .TimesSeconds = std::move(ch->TimesSeconds),
                    .Values = std::move(ch->Values),
                });
            }
            if (!resolved_clip.Channels.empty()) {
                if (auto *existing = R.try_get<MorphWeightAnimation>(instance_entity)) {
                    existing->Clips.emplace_back(std::move(resolved_clip));
                } else {
                    R.emplace<MorphWeightAnimation>(instance_entity, MorphWeightAnimation{.Clips = {std::move(resolved_clip)}});
                }
            }
        }
    }

    { // Get timeline range from imported animation durations
        float max_dur = 0;
        for (const auto [_, anim] : R.view<const ArmatureAnimation>().each()) {
            for (const auto &clip : anim.Clips) max_dur = std::max(max_dur, clip.DurationSeconds);
        }
        for (const auto [_, anim] : R.view<const MorphWeightAnimation>().each()) {
            for (const auto &clip : anim.Clips) max_dur = std::max(max_dur, clip.DurationSeconds);
        }
        if (max_dur > 0) R.patch<AnimationTimeline>(SceneEntity, [&](auto &tl) { tl.EndFrame = int(std::ceil(max_dur * tl.Fps)); });
    }

    if (scene->ImageBasedLight) {
        auto scene_world = CreateIblFromExtIbl(Vk, GetBufferCtx(Buffers.get()), *CommandPool, *OneShotFence, *Slots, scene->Images, *scene->ImageBasedLight);
        if (!scene_world) {
            std::cerr << std::format("Warning: Failed to import EXT_lights_image_based scene world from '{}': {}\n", path.string(), scene_world.error());
        } else {
            auto &environments = *Environments;
            if (environments.ImportedSceneWorld) {
                ReleaseCubeSamplerSlot(*Slots, environments.ImportedSceneWorld->DiffuseEnv.SamplerSlot);
                ReleaseCubeSamplerSlot(*Slots, environments.ImportedSceneWorld->SpecularEnv.SamplerSlot);
            }
            environments.ImportedSceneWorld = std::move(*scene_world);
            const auto &pre = *environments.ImportedSceneWorld;
            environments.SceneWorld = {.Ibl = MakeIblSamplers(pre, environments), .Name = pre.Name};
        }
    }

    const auto selected_entity =
        first_mesh_object_entity != entt::null ? first_mesh_object_entity :
        first_armature_entity != entt::null    ? first_armature_entity :
        first_root_empty_entity != entt::null  ? first_root_empty_entity :
                                                 first_object_entity;
    if (selected_entity != entt::null) Select(selected_entity);
    import_rollback_guard.Enabled = false;

    return std::pair{first_mesh_entity, selected_entity};
}
