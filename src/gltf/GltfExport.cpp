#include "GltfConvert.h"
#include "GltfScene.h"
#include "project/Assets.h"
#include "render/LightComponents.h"

#include "File.h"
#include "Profile.h"
#include "TransformMath.h"
#include "Variant.h"
#include "animation/AnimationData.h"
#include "animation/Fields.h"
#include "animation/MorphWeights.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "audio/AcousticMaterial.h"
#include "audio/AudioSystem.h"
#include "audio/AudioTypes.h"
#include "audio/ContactModel.h"
#include "audio/ContactSurface.h"
#include "audio/ModalModes.h"
#include "gltf/AnimationPointers.h"
#include "image/ImageEncode.h"
#include "mesh/MeshAttributes.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "physics/PhysicsTypes.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "scene/CameraLens.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "scene/WorldTransform.h"
#include "selection/Selection.h"

#include "metal/MetalContext.h"
#include "state/Scene.h"
#include <fastgltf/base64.hpp>
#include <fastgltf/core.hpp>
#include <iostream>

#include <bit>
#include <cstring>
#include <map>

namespace gltf {
using namespace detail;
namespace {
std::optional<std::string> EmitExtras(size_t idx, fastgltf::Category cat, void *userPtr) {
    if (!userPtr) return std::nullopt;
    const auto &m = *static_cast<const ExtrasMap *>(userPtr);
    if (const auto it = m.find(ExtrasKey(uint32_t(cat), idx)); it != m.end()) return it->second;
    return std::nullopt;
}

// Encode tightly-packed RGBA8 pixels to the container for `mime`, dispatching to the generic encoders.
// KTX2 and DDS aren't supported and return an error.
std::expected<std::vector<std::byte>, std::string>
EncodeImageRgba8ForMime(MimeType mime, std::span<const std::byte> rgba8, uint32_t width, uint32_t height, int jpeg_quality, std::string_view name) {
    using enum MimeType;
    switch (mime) {
        case PNG: return EncodeImagePngRgba8(rgba8, width, height, name);
        case JPEG: return EncodeImageJpegRgba8(rgba8, width, height, jpeg_quality, name);
        case WEBP: return EncodeImageWebpRgba8(rgba8, width, height, name);
        case KTX2: return std::unexpected{std::format("KTX2 encoding not supported for image '{}' (no basisu encoder vendored).", name)};
        case DDS: return std::unexpected{std::format("DDS encoding not supported for image '{}'.", name)};
        case GltfBuffer:
        case OctetStream:
        case None: return std::unexpected{std::format("Unrecognized mime type for image '{}'.", name)};
    }
    return std::unexpected{std::format("Unhandled mime type for image '{}'.", name)};
}
// fastgltf's name fields are pmr::string (doesn't implicit-copy from std::string).
using FgString = std::remove_cvref_t<decltype(fastgltf::Material::name)>;
FgString ToFgStr(std::string_view s) { return FgString{s}; }

// fastgltf::Optional<T> doesn't implicit-convert from std::optional<U>.
template<typename T, typename U>
fastgltf::Optional<T> ToFgOpt(const std::optional<U> &o) { return o ? fastgltf::Optional<T>{T(*o)} : fastgltf::Optional<T>{}; }
template<typename T, typename U, typename Fn>
fastgltf::Optional<T> ToFgOpt(const std::optional<U> &o, Fn &&fn) { return o ? fastgltf::Optional<T>{fn(*o)} : fastgltf::Optional<T>{}; }

std::optional<fastgltf::AccessorBoundsArray> MakeBounds(std::initializer_list<double> vals) {
    auto arr = fastgltf::AccessorBoundsArray::ForType<double>(vals.size());
    size_t i = 0;
    for (const double v : vals) arr.set<double>(i++, v);
    return arr;
}

// Append bytes, pad to 4-byte alignment, return starting offset.
uint32_t AppendAligned(std::vector<std::byte> &buffer, const std::byte *data, uint32_t size) {
    const uint32_t offset = buffer.size();
    buffer.insert(buffer.end(), data, data + size);
    while (buffer.size() % 4 != 0) buffer.emplace_back(std::byte{0});
    return offset;
}

template<typename T>
uint32_t AppendAligned(std::vector<std::byte> &buffer, std::span<const T> data) {
    return AppendAligned(buffer, reinterpret_cast<const std::byte *>(data.data()), data.size() * sizeof(T));
}

// Copies one field from each strided element into a four-byte-aligned binary blob without an intermediate vector.
// Returns the starting byte offset.
template<typename T, typename V>
uint32_t AppendField(std::vector<std::byte> &buffer, std::span<const V> data, T V::*field) {
    const uint32_t offset = buffer.size();
    buffer.resize(offset + data.size() * sizeof(T));
    auto *out = reinterpret_cast<T *>(buffer.data() + offset);
    for (size_t i = 0; i < data.size(); ++i) out[i] = data[i].*field;
    while (buffer.size() % 4 != 0) buffer.emplace_back(std::byte{0});
    return offset;
}

// Emits KHR_texture_transform when the UV transform is non-default or the source declared the extension, and sets `used` on emission.
std::unique_ptr<fastgltf::TextureTransform> MakeTextureTransform(const ::TextureInfo &ti, const TextureTransformMeta *meta, bool &used) {
    const bool has_transform = ti.UvOffset.x != 0.f || ti.UvOffset.y != 0.f ||
        ti.UvScale.x != 1.f || ti.UvScale.y != 1.f ||
        ti.UvRotation != 0.f;
    const bool source_had_ext = meta && meta->SourceHadExtension;
    if (!has_transform && !source_had_ext) return nullptr;
    used = true;
    auto t = std::make_unique<fastgltf::TextureTransform>();
    t->rotation = ti.UvRotation;
    t->uvOffset = std::bit_cast<fastgltf::math::nvec2>(ti.UvOffset);
    t->uvScale = std::bit_cast<fastgltf::math::nvec2>(ti.UvScale);
    if (meta && meta->SourceTexCoordOverride) t->texCoordIndex = *meta->SourceTexCoordOverride;
    return t;
}

// With meta, the parent texCoord and the extension's override are emitted separately.
void FillTextureInfo(fastgltf::TextureInfo &out, const ::TextureInfo &ti, const TextureTransformMeta *meta, bool &transform_used) {
    out.textureIndex = ti.Slot;
    out.texCoordIndex = meta ? meta->SourceBaseTexCoord : ti.TexCoord;
    out.transform = MakeTextureTransform(ti, meta, transform_used);
}
fastgltf::Optional<fastgltf::TextureInfo> ToFgTexInfo(const ::TextureInfo &ti, const TextureTransformMeta *meta, bool &transform_used) {
    if (ti.Slot == InvalidSlot) return {};
    fastgltf::TextureInfo out;
    FillTextureInfo(out, ti, meta, transform_used);
    return fastgltf::Optional<fastgltf::TextureInfo>{std::move(out)};
}
fastgltf::Optional<fastgltf::NormalTextureInfo> ToFgNormalTexInfo(const ::TextureInfo &ti, float scale, const TextureTransformMeta *meta, bool &transform_used) {
    if (ti.Slot == InvalidSlot) return {};
    fastgltf::NormalTextureInfo out;
    FillTextureInfo(out, ti, meta, transform_used);
    out.scale = scale;
    return fastgltf::Optional<fastgltf::NormalTextureInfo>{std::move(out)};
}
fastgltf::Optional<fastgltf::OcclusionTextureInfo> ToFgOcclusionTexInfo(const ::TextureInfo &ti, float strength, const TextureTransformMeta *meta, bool &transform_used) {
    if (ti.Slot == InvalidSlot) return {};
    fastgltf::OcclusionTextureInfo out;
    FillTextureInfo(out, ti, meta, transform_used);
    out.strength = strength;
    return fastgltf::Optional<fastgltf::OcclusionTextureInfo>{std::move(out)};
}

fastgltf::Camera ConvertCameraToFg(const CameraLens &cam, std::string_view name) {
    auto camera = std::visit(
        [](const auto &proj) -> std::variant<fastgltf::Camera::Perspective, fastgltf::Camera::Orthographic> {
            using P = std::decay_t<decltype(proj)>;
            if constexpr (std::is_same_v<P, Perspective>) {
                return fastgltf::Camera::Perspective{
                    .aspectRatio = proj.HasAspectRatio() ? fastgltf::Optional<fastgltf::num>{proj.AspectRatio} : fastgltf::Optional<fastgltf::num>{},
                    .yfov = proj.FieldOfViewRad,
                    .zfar = proj.HasFarClip() ? fastgltf::Optional<fastgltf::num>{proj.FarClip} : fastgltf::Optional<fastgltf::num>{},
                    .znear = proj.NearClip,
                };
            } else {
                return fastgltf::Camera::Orthographic{
                    .xmag = proj.Mag.x,
                    .ymag = proj.Mag.y,
                    .zfar = proj.FarClip,
                    .znear = proj.NearClip,
                };
            }
        },
        cam
    );
    return fastgltf::Camera{.camera = std::move(camera), .name = ToFgStr(name)};
}

// Writes each channel's rest value over the field it animates in `record`, so the export holds authored values.
void OverlayRest(const state::Scene &r, state::Entity e, state::TypeKey component, uint16_t index, std::span<std::byte> record) {
    const auto *clips = r.try_get<const AnimationClips>(e);
    if (!clips) return;
    for (const auto &clip : clips->Clips)
        for (const auto &channel : clip.Channels)
            if (channel.Target.Component == component && channel.Target.Index == index) std::memcpy(record.data() + channel.Target.Offset, channel.Rest.data(), channel.Rest.size() * sizeof(float));
}

fastgltf::Light ConvertLightToFg(const PunctualLight &pl, std::string_view name) {
    const auto type = pl.Type == PunctualLightType::Point ? fastgltf::LightType::Point : pl.Type == PunctualLightType::Spot ? fastgltf::LightType::Spot :
                                                                                                                              fastgltf::LightType::Directional;
    const bool is_spot = type == fastgltf::LightType::Spot;
    return fastgltf::Light{
        .type = type,
        .color = std::bit_cast<fastgltf::math::nvec3>(pl.Color),
        .intensity = pl.Intensity,
        .range = (type != fastgltf::LightType::Directional && pl.Range > 0) ? fastgltf::Optional<fastgltf::num>{pl.Range} : fastgltf::Optional<fastgltf::num>{},
        .innerConeAngle = is_spot ? fastgltf::Optional<fastgltf::num>{pl.InnerConeAngle} : fastgltf::Optional<fastgltf::num>{},
        .outerConeAngle = is_spot ? fastgltf::Optional<fastgltf::num>{pl.OuterConeAngle} : fastgltf::Optional<fastgltf::num>{},
        .name = ToFgStr(name),
    };
}

} // namespace

std::expected<void, std::string> SaveGltf(const std::filesystem::path &path, const state::Scene &r, state::Entity viewport, SaveOptions options) {
    const profile::CpuScope scope{"SaveGltf"};
    const auto &meshes = r.ctx().get<const MeshStore>();
    const auto &buffers = r.ctx().get<const GpuBuffers>();

    // Entities of `view` in the source order `index_of` reads, with runtime-added ones after the source range in view order.
    const auto ordered_by_source = [&](auto view, auto &&index_of) {
        std::vector<state::Entity> ordered;
        for (const auto e : view) ordered.emplace_back(e);
        std::ranges::stable_sort(ordered, {}, [&](state::Entity e) { return index_of(e).value_or(std::numeric_limits<uint32_t>::max()); });
        return ordered;
    };
    const auto source_index = [&](state::Entity e) -> std::optional<uint32_t> {
        const auto *index = r.try_get<const SourceIndex>(e);
        return index ? std::optional{index->Value} : std::nullopt;
    };

    // Read source metadata and texture, image, and sampler arrays from gltf::SourceAssets.
    // Encoded images, sampler details, and asset metadata cannot be reconstructed from registry or GPU state.
    // Emit cameras and lights from entity components and materials from PBRMaterial plus MaterialSourceMeta.
    const auto *src_assets = r.try_get<const gltf::SourceAssets>(viewport);
    // Read source-form scene metadata directly from src_assets at each emission site.
    static const gltf::SourceAssets EmptySourceAssets{};
    const auto &sa = src_assets ? *src_assets : EmptySourceAssets;
    const auto &names = r.ctx().get<const MaterialStore>().Names;
    const auto material_count = buffers.Materials.Count<PBRMaterial>();
    const auto &material_metas = src_assets ? src_assets->MaterialMetas : std::vector<MaterialSourceMeta>{};

    // Preserve source mesh ordering through the source layout.
    // Append runtime-created meshes after the source range.
    std::unordered_map<state::Entity, uint32_t> mesh_entity_to_index;
    uint32_t mesh_count = 0;
    for (const auto [e, _, layout] : r.view<const MeshHandle, const MeshSourceLayout>().each()) {
        mesh_entity_to_index[e] = layout.Index;
        mesh_count = std::max(mesh_count, layout.Index + 1u);
    }
    for (const auto e : r.view<const MeshHandle>()) {
        if (!mesh_entity_to_index.contains(e)) mesh_entity_to_index[e] = mesh_count++;
    }

    // Group triangle, line, and point entities by mesh index.
    // The emit pass reads vertex, face, skin, and morph data directly from MeshStore.
    struct MeshEntitySet {
        state::Entity Triangles{state::Null}, Lines{state::Null}, Points{state::Null};
        std::string Name;
    };
    std::vector<MeshEntitySet> mesh_groups(mesh_count);
    for (const auto &[entity, idx] : mesh_entity_to_index) {
        auto &g = mesh_groups[idx];
        const auto *layout = r.try_get<const MeshSourceLayout>(entity);
        const auto k = layout ? layout->Kind : MeshKind::Triangles;
        if (k == MeshKind::Triangles) g.Triangles = entity;
        else if (k == MeshKind::Lines) g.Lines = entity;
        else g.Points = entity;
        if (g.Name.empty() && layout) g.Name = layout->Name;
    }

    // Cameras and lights emit one resource per component-bearing entity in source order.
    // Khronos samples do not share source cameras or lights across nodes.
    const auto node_of = [&](auto member) {
        return [&, member](state::Entity e) { const auto *node = r.try_get<const GltfNode>(e); return node ? node->*member : std::nullopt; };
    };
    std::vector<state::Entity> camera_entities;
    for (const auto e : r.view<const Perspective>()) camera_entities.emplace_back(e);
    for (const auto e : r.view<const Orthographic>()) camera_entities.emplace_back(e);
    const auto camera_entities_ordered = ordered_by_source(camera_entities, node_of(&GltfNode::Camera));
    const auto light_entities_ordered = ordered_by_source(r.view<const PunctualLight>(), node_of(&GltfNode::Light));
    std::unordered_map<state::Entity, uint32_t> camera_entity_to_index, light_entity_to_index;
    for (uint32_t i = 0; i < camera_entities_ordered.size(); ++i) camera_entity_to_index[camera_entities_ordered[i]] = i;
    for (uint32_t i = 0; i < light_entities_ordered.size(); ++i) light_entity_to_index[light_entities_ordered[i]] = i;

    // One entry per emitted node: live source nodes compacted to [0, k) in source order, then runtime-created objects as scene roots, then offset-collider children.
    // Source parent links restore the imported hierarchy after runtime reparenting. A deleted source parent drops the link, so the child emits as a root.
    struct ExportNode {
        // The entity whose transform, name, physics, and audio represent the node: the lowest entity id among those sharing its source index.
        state::Entity Entity{state::Null};
        // Object entities sharing the node in ascending entity id order, for EXT_mesh_gpu_instancing.
        std::vector<state::Entity> Instances;
        std::optional<uint32_t> Parent;
        // Dense child indices in source sibling order, offset-collider children last.
        std::vector<uint32_t> Children;
        // Set on the owner of a synthetic offset-collider child and on that child.
        std::optional<uint32_t> OffsetChild;
        state::Entity OffsetOwner{state::Null};
        bool Hidden{};
    };
    std::vector<ExportNode> nodes;
    std::unordered_map<state::Entity, uint32_t> node_index_of;
    {
        std::map<uint32_t, std::vector<state::Entity>> entities_by_source;
        for (const auto [e, node] : r.view<const GltfNode>().each()) {
            if (node.Index) entities_by_source[*node.Index].emplace_back(e);
        }
        std::unordered_map<uint32_t, uint32_t> source_to_dense;
        for (auto &[source, entities] : entities_by_source) {
            std::ranges::sort(entities);
            const auto ni = uint32_t(nodes.size());
            source_to_dense[source] = ni;
            auto &node = nodes.emplace_back();
            node.Entity = entities.front();
            for (const auto e : entities) {
                node_index_of[e] = ni;
                const auto *kind = r.try_get<const ObjectKind>(e);
                if (kind && kind->Value != ObjectType::Armature && r.all_of<Transform>(e)) node.Instances.emplace_back(e);
            }
        }
        for (const auto [e, _t, kind] : r.view<const Transform, const ObjectKind>().each()) {
            if (kind.Value == ObjectType::Armature || node_index_of.contains(e)) continue; // Armatures round-trip via skins.
            node_index_of[e] = uint32_t(nodes.size());
            auto &node = nodes.emplace_back();
            node.Entity = e;
            node.Instances = {e};
        }
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> children(nodes.size());
        for (uint32_t ni = 0; ni < nodes.size(); ++ni) {
            const auto *node = r.try_get<const GltfNode>(nodes[ni].Entity);
            if (!node || !node->Parent) continue;
            const auto pit = source_to_dense.find(*node->Parent);
            if (pit == source_to_dense.end()) continue;
            nodes[ni].Parent = pit->second;
            children[pit->second].emplace_back(node->Sibling.value_or(ni), ni);
        }
        // A collider with a local offset emits its geometry on a synthetic child node carrying the offset.
        for (uint32_t ni = 0, count = uint32_t(nodes.size()); ni < count; ++ni) {
            const auto *cs = r.try_get<const ColliderShape>(nodes[ni].Entity);
            if (!cs || cs->LocalOffset == vec3{0}) continue;
            const auto child = uint32_t(nodes.size());
            nodes[ni].OffsetChild = child;
            auto &node = nodes.emplace_back();
            node.Parent = ni;
            node.OffsetOwner = nodes[ni].Entity;
            children.emplace_back();
            children[ni].emplace_back(std::numeric_limits<uint32_t>::max(), child);
        }
        for (uint32_t ni = 0; ni < nodes.size(); ++ni) {
            std::ranges::sort(children[ni], {}, &std::pair<uint32_t, uint32_t>::first);
            for (const auto &[_, child] : children[ni]) nodes[ni].Children.emplace_back(child);
        }
    }

    // Scenes to emit, in source order. active_scene is the default.
    std::vector<state::Entity> scenes_ordered;
    for (const auto e : r.view<const Scene>()) scenes_ordered.emplace_back(e);
    std::ranges::sort(scenes_ordered, {}, [&](state::Entity e) { return source_index(e).value_or(std::numeric_limits<uint32_t>::max()); });
    state::Entity active_scene = state::Null;
    for (const auto e : r.view<const ActiveScene>()) active_scene = e;

    // Returns live nodes without a live source parent, restricted by SceneMembership when scene is nonnull.
    // Sorts roots because glTF scene-node order is nonsemantic.
    const auto compute_roots = [&](state::Entity scene) {
        std::vector<uint32_t> roots;
        for (uint32_t ni = 0; ni < nodes.size(); ++ni) {
            const auto entity = nodes[ni].Entity;
            if (entity == state::Null || nodes[ni].Parent) continue;
            if (scene != state::Null) {
                const auto *sm = r.try_get<const SceneMembership>(entity);
                if (sm && std::ranges::find(sm->Scenes, scene) == sm->Scenes.end()) continue;
            }
            roots.emplace_back(ni);
        }
        return roots;
    };

    // KHR_node_visibility: a node is hidden iff it has no RenderInstance and every child is hidden too.
    // Stubs are unreachable from scene roots, so they default to not-hidden and emit no spurious visible:false.
    // A post-order walk from the roots fills Hidden so parents can check children without recursion or multi-pass.
    // Nodes only in non-active scenes count as not-hidden, since their missing RenderInstance is a switch-time artifact, not a user-set hide.
    {
        const auto node_in_active_scene = [&](state::Entity entity) {
            const auto *sm = entity != state::Null ? r.try_get<const SceneMembership>(entity) : nullptr;
            return !sm || std::ranges::find(sm->Scenes, active_scene) != sm->Scenes.end();
        };
        const auto dfs = [&](this const auto &self, uint32_t ni) -> bool {
            auto &node = nodes[ni];
            bool hidden = node.Entity != state::Null && node_in_active_scene(node.Entity) && !r.all_of<RenderInstance>(node.Entity);
            for (const auto child : node.Children) {
                if (!self(child)) hidden = false;
            }
            node.Hidden = hidden;
            return hidden;
        };
        for (const auto se : scenes_ordered) {
            for (const auto ni : compute_roots(se)) dfs(ni);
        }
        if (scenes_ordered.empty()) {
            for (const auto ni : compute_roots(state::Null)) dfs(ni);
        }
    }

    // Construct the asset before allocator-bound collision filters.
    fastgltf::Asset asset;

    // Emit source-aligned physics resources directly into the asset.
    std::unordered_map<state::Entity, uint32_t> physics_material_to_index, physics_jointdef_to_index, collision_filter_to_index;
    {
        auto mat_view = r.view<const PhysicsMaterial>();
        for (const auto e : ordered_by_source(mat_view, source_index)) {
            const auto &pm = mat_view.get<const PhysicsMaterial>(e);
            physics_material_to_index[e] = asset.physicsMaterials.size();
            asset.physicsMaterials.emplace_back(fastgltf::PhysicsMaterial{
                .staticFriction = pm.StaticFriction,
                .dynamicFriction = pm.DynamicFriction,
                .restitution = pm.Restitution,
                .frictionCombine = FromCombineMode(pm.FrictionCombine),
                .restitutionCombine = FromCombineMode(pm.RestitutionCombine),
            });
        }
        auto jd_view = r.view<const ::PhysicsJointDef>();
        for (const auto e : ordered_by_source(jd_view, source_index)) {
            const auto &jd = jd_view.get<const ::PhysicsJointDef>(e);
            fastgltf::pmr::MaybeSmallVector<fastgltf::JointLimit> limits;
            limits.reserve(jd.Limits.size());
            for (const auto &lim : jd.Limits) {
                fastgltf::pmr::SmallVector<uint8_t, 3> linear_axes;
                for (const auto a : lim.LinearAxes) linear_axes.emplace_back(a);
                fastgltf::pmr::SmallVector<uint8_t, 3> angular_axes;
                for (const auto a : lim.AngularAxes) angular_axes.emplace_back(a);
                limits.emplace_back(fastgltf::JointLimit{
                    .linearAxes = std::move(linear_axes),
                    .angularAxes = std::move(angular_axes),
                    .min = ToFgOpt<fastgltf::num>(lim.Min),
                    .max = ToFgOpt<fastgltf::num>(lim.Max),
                    .stiffness = ToFgOpt<fastgltf::num>(lim.Stiffness),
                    .damping = lim.Damping,
                });
            }
            fastgltf::pmr::MaybeSmallVector<fastgltf::JointDrive> drives;
            drives.reserve(jd.Drives.size());
            for (const auto &drv : jd.Drives) {
                drives.emplace_back(fastgltf::JointDrive{
                    .type = drv.Type == PhysicsDriveType::Angular ? fastgltf::DriveType::Angular : fastgltf::DriveType::Linear,
                    .mode = drv.Mode == PhysicsDriveMode::Acceleration ? fastgltf::DriveMode::Acceleration : fastgltf::DriveMode::Force,
                    .axis = drv.Axis,
                    .maxForce = drv.MaxForce,
                    .positionTarget = drv.PositionTarget,
                    .velocityTarget = drv.VelocityTarget,
                    .stiffness = drv.Stiffness,
                    .damping = drv.Damping,
                });
            }
            physics_jointdef_to_index[e] = asset.physicsJoints.size();
            asset.physicsJoints.emplace_back(fastgltf::PhysicsJoint{.limits = std::move(limits), .drives = std::move(drives)});
        }
        const auto resolve_system_names = [&](std::span<const state::Entity> systems) {
            fastgltf::pmr::MaybeSmallVector<FgString> out;
            out.reserve(systems.size());
            for (const auto se : systems) {
                if (const auto *cs = r.try_get<const CollisionSystem>(se)) out.emplace_back(ToFgStr(cs->Name));
            }
            return out;
        };
        auto cf_view = r.view<const CollisionFilter>();
        for (const auto e : ordered_by_source(cf_view, source_index)) {
            const auto &f = cf_view.get<const CollisionFilter>(e);
            collision_filter_to_index[e] = asset.collisionFilters.size();
            fastgltf::CollisionFilter out{.collisionSystems = resolve_system_names(f.Systems), .notCollideWithSystems = {}, .collideWithSystems = {}};
            if (f.Mode == CollideMode::Allowlist) out.collideWithSystems = resolve_system_names(f.CollideSystems);
            else if (f.Mode == CollideMode::Blocklist) out.notCollideWithSystems = resolve_system_names(f.CollideSystems);
            asset.collisionFilters.emplace_back(std::move(out));
        }
    }

    // Empty strings are omitted by fastgltf's writer.
    asset.assetInfo = fastgltf::AssetInfo{
        .gltfVersion = "2.0",
        .minVersion = ToFgStr(sa.MinVersion),
        .copyright = ToFgStr(sa.Copyright),
        .generator = ToFgStr(sa.Generator),
        .extras = ToFgStr(sa.AssetExtras),
        .extensions = ToFgStr(sa.AssetExtensions),
    };

    std::vector<std::byte> bin;
    std::vector<fastgltf::BufferView> bufferViews;
    std::vector<fastgltf::Accessor> accessors;

    auto AddBufferView = [&](uint32_t offset, uint32_t length, std::optional<uint32_t> stride = {}, std::optional<fastgltf::BufferTarget> target = {}) {
        bufferViews.emplace_back(fastgltf::BufferView{
            .bufferIndex = 0,
            .byteOffset = offset,
            .byteLength = length,
            .byteStride = ToFgOpt<size_t>(stride),
            .target = ToFgOpt<fastgltf::BufferTarget>(target),
            .meshoptCompression = nullptr,
            .name = {},
        });
        return bufferViews.size() - 1;
    };

    auto AddAccessor = [&](uint32_t bufferViewIdx, uint32_t count, fastgltf::AccessorType type, fastgltf::ComponentType component,
                           std::optional<fastgltf::AccessorBoundsArray> min = {}, std::optional<fastgltf::AccessorBoundsArray> max = {}) {
        accessors.emplace_back(fastgltf::Accessor{
            .byteOffset = 0,
            .count = count,
            .type = type,
            .componentType = component,
            .normalized = false,
            .max = std::move(max),
            .min = std::move(min),
            .bufferViewIndex = bufferViewIdx,
            .sparse = {},
            .name = {},
        });
        return accessors.size() - 1;
    };

    const auto AddDataAccessor = [&]<typename T>(std::span<const T> data, fastgltf::AccessorType type, fastgltf::ComponentType component, std::optional<fastgltf::BufferTarget> target = {}) {
        const uint32_t off = AppendAligned<T>(bin, data);
        const uint32_t bv = AddBufferView(off, data.size() * sizeof(T), {}, target);
        return AddAccessor(bv, data.size(), type, component);
    };

    const auto AddVec3Accessor = [&](std::span<const vec3> data, bool with_bounds, fastgltf::BufferTarget target) {
        const uint32_t vcount = data.size();
        if (!with_bounds || vcount == 0) return AddDataAccessor(data, fastgltf::AccessorType::Vec3, fastgltf::ComponentType::Float, target);
        vec3 lo = data[0], hi = data[0];
        for (const auto &p : data) {
            lo = numeric::Min(lo, p);
            hi = numeric::Max(hi, p);
        }
        const uint32_t off = AppendAligned<vec3>(bin, data);
        const uint32_t bv = AddBufferView(off, vcount * sizeof(vec3), {}, target);
        return AddAccessor(
            bv, vcount, fastgltf::AccessorType::Vec3, fastgltf::ComponentType::Float,
            MakeBounds({lo.x, lo.y, lo.z}), MakeBounds({hi.x, hi.y, hi.z})
        );
    };

    // Write a strided field directly into one attribute accessor.
    const auto AddFieldAccessor = [&]<typename T, typename V>(std::span<const V> data, T V::*field, fastgltf::AccessorType type, fastgltf::BufferTarget target) {
        const uint32_t off = AppendField<T>(bin, data, field);
        const uint32_t bv = AddBufferView(off, data.size() * sizeof(T), {}, target);
        return AddAccessor(bv, data.size(), type, fastgltf::ComponentType::Float);
    };
    const auto AddPositionFieldAccessor = [&]<typename V>(std::span<const V> data, vec3 V::*field, fastgltf::BufferTarget target) {
        const uint32_t vcount = data.size();
        if (vcount == 0) return AddFieldAccessor.template operator()<vec3>(data, field, fastgltf::AccessorType::Vec3, target);
        vec3 lo = data[0].*field, hi = lo;
        for (const auto &v : data) {
            lo = numeric::Min(lo, v.*field);
            hi = numeric::Max(hi, v.*field);
        }
        const uint32_t off = AppendField<vec3>(bin, data, field);
        const uint32_t bv = AddBufferView(off, vcount * sizeof(vec3), {}, target);
        return AddAccessor(
            bv, vcount, fastgltf::AccessorType::Vec3, fastgltf::ComponentType::Float,
            MakeBounds({lo.x, lo.y, lo.z}), MakeBounds({hi.x, hi.y, hi.z})
        );
    };

    // Emit COLOR_0 from gathered per-corner values, preserving source component count.
    const auto EmitColor0Values = [&](fastgltf::pmr::SmallVector<fastgltf::Attribute, 4> &out, std::span<const vec4> colors, uint8_t component_count) {
        if (component_count == 3) {
            const uint32_t vcount = colors.size();
            const uint32_t off = bin.size();
            bin.resize(off + vcount * sizeof(vec3));
            auto *outp = reinterpret_cast<vec3 *>(bin.data() + off);
            for (uint32_t i = 0; i < vcount; ++i) outp[i] = {colors[i].x, colors[i].y, colors[i].z};
            while (bin.size() % 4 != 0) bin.emplace_back(std::byte{0});
            const uint32_t bv = AddBufferView(off, vcount * sizeof(vec3), {}, fastgltf::BufferTarget::ArrayBuffer);
            out.emplace_back(fastgltf::Attribute{"COLOR_0", AddAccessor(bv, vcount, fastgltf::AccessorType::Vec3, fastgltf::ComponentType::Float)});
        } else {
            out.emplace_back(fastgltf::Attribute{"COLOR_0", AddDataAccessor(colors, fastgltf::AccessorType::Vec4, fastgltf::ComponentType::Float, fastgltf::BufferTarget::ArrayBuffer)});
        }
    };

    asset.samplers.reserve(sa.Samplers.size());
    for (const auto &s : sa.Samplers) {
        asset.samplers.emplace_back(fastgltf::Sampler{
            .magFilter = ToFgOpt<fastgltf::Filter>(s.MagFilter, FromFilter),
            .minFilter = ToFgOpt<fastgltf::Filter>(s.MinFilter, FromFilter),
            .wrapS = FromWrap(s.WrapS),
            .wrapT = FromWrap(s.WrapT),
            .name = ToFgStr(s.Name),
        });
    }

    // Each image emits in its source form: an external URI, a data URI, or a buffer view. A dirty image re-encodes from its GPU texture first.
    asset.images.reserve(sa.Images.size());
    std::unordered_map<uint32_t, const TextureEntry *> texture_for_image;
    for (const auto &tex : r.ctx().get<const TextureStore>().Textures) {
        if (tex.SourceImageIndex != UINT32_MAX) texture_for_image.emplace(tex.SourceImageIndex, &tex);
    }
    const auto reencode_from_gpu = [&](uint32_t img_idx, gltf::MimeType target, std::string_view name)
        -> std::expected<std::pair<std::vector<std::byte>, gltf::MimeType>, std::string> {
        const auto it = texture_for_image.find(img_idx);
        if (it == texture_for_image.end()) return std::unexpected{std::format("Image '{}' has no GPU texture; cannot re-encode.", name)};
        const auto *ctx = r.ctx().find<const mtl::Context>();
        if (!ctx) return std::unexpected{std::format("Image '{}' needs GPU readback but no Metal context is registered.", name)};
        auto rgba8 = ReadbackTextureRgba8(*ctx, *it->second);
        if (!rgba8) return std::unexpected{std::move(rgba8.error())};
        const auto w = it->second->Image.Extent.Width, h = it->second->Image.Extent.Height;
        if (auto enc = EncodeImageRgba8ForMime(target, *rgba8, w, h, options.LossyImageQuality, name)) {
            return std::pair{std::move(*enc), target};
        } else if (target == gltf::MimeType::PNG) {
            return std::unexpected{std::move(enc.error())};
        } else {
            // PNG provides the encoder fallback.
            std::cerr << std::format("Warning: image '{}': {}. Falling back to PNG.\n", name, enc.error());
            auto png = EncodeImagePngRgba8(*rgba8, w, h, name);
            if (!png) return std::unexpected{std::move(png.error())};
            return std::pair{std::move(*png), gltf::MimeType::PNG};
        }
    };
    for (uint32_t i = 0; i < sa.Images.size(); ++i) {
        const auto &img = sa.Images[i];
        using Source = gltf::Image::SourceKind;
        const bool encodable = img.MimeType != gltf::MimeType::KTX2 && img.MimeType != gltf::MimeType::DDS;
        auto form = img.Source;
        auto mime = img.MimeType;
        std::vector<std::byte> owned;
        std::span<const std::byte> bytes = img.Bytes;
        if (img.IsDirty && encodable) {
            auto re = reencode_from_gpu(i, img.MimeType, img.Name);
            if (!re) return std::unexpected{std::move(re.error())};
            owned = std::move(re->first);
            mime = re->second;
            bytes = owned;
            // The re-encoded bytes no longer match the external file, so they embed.
            if (form == Source::External) form = Source::Embedded;
        } else if (img.IsDirty) {
            // KTX2 and DDS lack an encoder, so their source bytes stand.
            std::cerr << std::format("Warning: image '{}' is dirty but {} re-encoding isn't supported; emitting original bytes.\n", img.Name, img.MimeType == gltf::MimeType::KTX2 ? "KTX2" : "DDS");
        }
        // A project archives external sources, so they embed on export.
        if (form == Source::External && project::Assets::IsReference(img.SourcePath)) form = Source::Embedded;
        if (form == Source::External) {
            std::error_code ec;
            if (!std::filesystem::is_regular_file(img.SourcePath, ec)) {
                if (auto re = reencode_from_gpu(i, gltf::MimeType::PNG, img.Name)) {
                    std::cerr << std::format("Warning: image '{}' source '{}' is missing; embedding as PNG.\n", img.Name, img.SourcePath);
                    owned = std::move(re->first);
                    mime = gltf::MimeType::PNG;
                    bytes = owned;
                    form = Source::Embedded;
                } else {
                    std::cerr << std::format("Warning: image '{}' source '{}' is missing and re-encoding failed ({}); emitting the URI as-is.\n", img.Name, img.SourcePath, re.error());
                }
            }
        }
        if (form != Source::External && bytes.empty() && !img.SourcePath.empty()) {
            auto file = File::Read(project::ResolveAsset(r, img.SourcePath));
            if (!file) return std::unexpected{std::move(file.error())};
            owned = std::move(*file);
            bytes = owned;
        }

        if (form == Source::External) {
            asset.images.emplace_back(fastgltf::Image{
                .data = fastgltf::sources::URI{.fileByteOffset = 0, .uri = fastgltf::URI{std::string_view{img.Uri}}, .mimeType = img.SourceHadMimeType ? FromMimeType(img.MimeType) : fastgltf::MimeType::None},
                .name = ToFgStr(img.Name),
            });
        } else if (form == Source::DataUri && !bytes.empty()) {
            const auto fg_mime = FromMimeType(mime);
            const auto mime_str = fg_mime == fastgltf::MimeType::None ? std::string{} : std::string{fastgltf::getMimeTypeString(fg_mime)};
            const auto data_uri = "data:" + mime_str + ";base64," + fastgltf::base64::encode(reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size());
            asset.images.emplace_back(fastgltf::Image{
                .data = fastgltf::sources::URI{.fileByteOffset = 0, .uri = fastgltf::URI{data_uri}, .mimeType = fastgltf::MimeType::None},
                .name = ToFgStr(img.Name),
            });
        } else {
            // An image without bytes keeps its required bufferView slot with one padding byte.
            const uint32_t bv = bytes.empty() ? AddBufferView(AppendAligned(bin, reinterpret_cast<const std::byte *>("\0"), 1), 1) : AddBufferView(AppendAligned(bin, bytes.data(), bytes.size()), bytes.size());
            asset.images.emplace_back(fastgltf::Image{
                .data = fastgltf::sources::BufferView{.bufferViewIndex = bv, .mimeType = FromMimeType(mime)},
                .name = ToFgStr(img.Name),
            });
        }
    }

    asset.textures.reserve(sa.Textures.size());
    for (const auto &t : sa.Textures) {
        asset.textures.emplace_back(fastgltf::Texture{
            .samplerIndex = ToFgOpt<size_t>(t.SamplerIndex),
            .imageIndex = ToFgOpt<size_t>(t.ImageIndex),
            .basisuImageIndex = ToFgOpt<size_t>(t.BasisuImageIndex),
            .ddsImageIndex = ToFgOpt<size_t>(t.DdsImageIndex),
            .webpImageIndex = ToFgOpt<size_t>(t.WebpImageIndex),
            .name = ToFgStr(t.Name),
        });
    }

    std::vector<fastgltf::Optional<size_t>> material_indices(material_count);
    asset.materials.reserve(material_count);
    bool uses_texture_transform = false;
    using M = MaterialSourceMeta;
    static const MaterialSourceMeta DefaultMeta{};
    for (uint32_t i = 1; i < material_count; ++i) {
        const auto source_idx = i - 1;
        auto pbr = buffers.Materials.GetSpan<PBRMaterial>()[i];
        OverlayRest(r, viewport, state::Key<MaterialStore>(), uint16_t(i), std::as_writable_bytes(std::span{&pbr, 1}));
        const auto &meta = source_idx < material_metas.size() ? material_metas[source_idx] : DefaultMeta;
        if (meta.ImplicitDefault) continue;
        material_indices[i] = asset.materials.size();
        const auto bits = meta.ExtensionPresence;
        for (uint32_t s = 0; s < MTS_Count; ++s) MaterialTextureSlots[s].Get(pbr).Slot = meta.TextureSlots[s];

        const std::string name = (!meta.NameWasEmpty && i < names.size()) ? names[i] : std::string{};
        fastgltf::Material out;
        out.name = ToFgStr(name);
        out.pbrData.baseColorFactor = std::bit_cast<fastgltf::math::nvec4>(pbr.BaseColorFactor);
        out.pbrData.metallicFactor = pbr.MetallicFactor;
        out.pbrData.roughnessFactor = pbr.RoughnessFactor;
        out.pbrData.baseColorTexture = ToFgTexInfo(pbr.BaseColorTexture, &meta.BaseSlotMeta[0], uses_texture_transform);
        out.pbrData.metallicRoughnessTexture = ToFgTexInfo(pbr.MetallicRoughnessTexture, &meta.BaseSlotMeta[1], uses_texture_transform);
        out.normalTexture = ToFgNormalTexInfo(pbr.NormalTexture, pbr.NormalScale, &meta.BaseSlotMeta[2], uses_texture_transform);
        out.occlusionTexture = ToFgOcclusionTexInfo(pbr.OcclusionTexture, pbr.OcclusionStrength, &meta.BaseSlotMeta[3], uses_texture_transform);
        out.emissiveTexture = ToFgTexInfo(pbr.EmissiveTexture, &meta.BaseSlotMeta[4], uses_texture_transform);
        out.emissiveFactor = std::bit_cast<fastgltf::math::nvec3>(pbr.EmissiveFactor);
        if (bits & M::ExtEmissiveStrength || pbr.EmissiveStrength != 1.f) out.emissiveStrength = fastgltf::Optional<fastgltf::num>{pbr.EmissiveStrength};
        out.alphaMode = FromAlphaMode(pbr.AlphaMode);
        out.alphaCutoff = pbr.AlphaCutoff;
        out.doubleSided = pbr.DoubleSided != 0u;
        out.unlit = pbr.Unlit != 0u;
        if (bits & M::ExtIor) out.ior = fastgltf::Optional<fastgltf::num>{pbr.Ior};
        if (bits & M::ExtDispersion) out.dispersion = fastgltf::Optional<fastgltf::num>{pbr.Dispersion};

        if (bits & M::ExtSheen) {
            out.sheen = std::make_unique<fastgltf::MaterialSheen>();
            out.sheen->sheenColorFactor = std::bit_cast<fastgltf::math::nvec3>(pbr.Sheen.ColorFactor);
            out.sheen->sheenRoughnessFactor = pbr.Sheen.RoughnessFactor;
            out.sheen->sheenColorTexture = ToFgTexInfo(pbr.Sheen.ColorTexture, nullptr, uses_texture_transform);
            out.sheen->sheenRoughnessTexture = ToFgTexInfo(pbr.Sheen.RoughnessTexture, nullptr, uses_texture_transform);
        }
        if (bits & M::ExtSpecular) {
            out.specular = std::make_unique<fastgltf::MaterialSpecular>();
            out.specular->specularFactor = pbr.Specular.Factor;
            out.specular->specularColorFactor = std::bit_cast<fastgltf::math::nvec3>(pbr.Specular.ColorFactor);
            out.specular->specularTexture = ToFgTexInfo(pbr.Specular.Texture, nullptr, uses_texture_transform);
            out.specular->specularColorTexture = ToFgTexInfo(pbr.Specular.ColorTexture, nullptr, uses_texture_transform);
        }
        if (bits & M::ExtTransmission) {
            out.transmission = std::make_unique<fastgltf::MaterialTransmission>();
            out.transmission->transmissionFactor = pbr.Transmission.Factor;
            out.transmission->transmissionTexture = ToFgTexInfo(pbr.Transmission.Texture, nullptr, uses_texture_transform);
        }
        if (bits & M::ExtDiffuseTransmission) {
            out.diffuseTransmission = std::make_unique<fastgltf::MaterialDiffuseTransmission>();
            out.diffuseTransmission->diffuseTransmissionFactor = pbr.DiffuseTransmission.Factor;
            out.diffuseTransmission->diffuseTransmissionColorFactor = std::bit_cast<fastgltf::math::nvec3>(pbr.DiffuseTransmission.ColorFactor);
            out.diffuseTransmission->diffuseTransmissionTexture = ToFgTexInfo(pbr.DiffuseTransmission.Texture, nullptr, uses_texture_transform);
            out.diffuseTransmission->diffuseTransmissionColorTexture = ToFgTexInfo(pbr.DiffuseTransmission.ColorTexture, nullptr, uses_texture_transform);
        }
        if (bits & M::ExtVolume) {
            out.volume = std::make_unique<fastgltf::MaterialVolume>();
            out.volume->thicknessFactor = pbr.Volume.ThicknessFactor;
            out.volume->attenuationColor = std::bit_cast<fastgltf::math::nvec3>(pbr.Volume.AttenuationColor);
            out.volume->attenuationDistance = pbr.Volume.AttenuationDistance > 0.f ? pbr.Volume.AttenuationDistance : std::numeric_limits<float>::infinity();
            out.volume->thicknessTexture = ToFgTexInfo(pbr.Volume.ThicknessTexture, nullptr, uses_texture_transform);
        }
        if (bits & M::ExtClearcoat) {
            out.clearcoat = std::make_unique<fastgltf::MaterialClearcoat>();
            out.clearcoat->clearcoatFactor = pbr.Clearcoat.Factor;
            out.clearcoat->clearcoatRoughnessFactor = pbr.Clearcoat.RoughnessFactor;
            out.clearcoat->clearcoatTexture = ToFgTexInfo(pbr.Clearcoat.Texture, nullptr, uses_texture_transform);
            out.clearcoat->clearcoatRoughnessTexture = ToFgTexInfo(pbr.Clearcoat.RoughnessTexture, nullptr, uses_texture_transform);
            out.clearcoat->clearcoatNormalTexture = ToFgNormalTexInfo(pbr.Clearcoat.NormalTexture, pbr.Clearcoat.NormalScale, nullptr, uses_texture_transform);
        }
        if (bits & M::ExtAnisotropy) {
            out.anisotropy = std::make_unique<fastgltf::MaterialAnisotropy>();
            out.anisotropy->anisotropyStrength = pbr.Anisotropy.Strength;
            out.anisotropy->anisotropyRotation = pbr.Anisotropy.Rotation;
            out.anisotropy->anisotropyTexture = ToFgTexInfo(pbr.Anisotropy.Texture, nullptr, uses_texture_transform);
        }
        if (bits & M::ExtIridescence) {
            out.iridescence = std::make_unique<fastgltf::MaterialIridescence>();
            out.iridescence->iridescenceFactor = pbr.Iridescence.Factor;
            out.iridescence->iridescenceIor = pbr.Iridescence.Ior;
            out.iridescence->iridescenceThicknessMinimum = pbr.Iridescence.ThicknessMinimum;
            out.iridescence->iridescenceThicknessMaximum = pbr.Iridescence.ThicknessMaximum;
            out.iridescence->iridescenceTexture = ToFgTexInfo(pbr.Iridescence.Texture, nullptr, uses_texture_transform);
            out.iridescence->iridescenceThicknessTexture = ToFgTexInfo(pbr.Iridescence.ThicknessTexture, nullptr, uses_texture_transform);
        }

        asset.materials.emplace_back(std::move(out));
    }

    // Animations: one glTF animation per scene animation with channels, in scene order.
    // Node transforms and morph weights emit classic channels. Other fields emit KHR_animation_pointer channels.
    bool uses_animation_pointer = false;
    {
        const auto *animations = r.try_get<const Animations>(viewport);
        const auto animation_names = animations ? std::span<const std::string>{animations->Names} : std::span<const std::string>{};
        asset.animations.reserve(animation_names.size());
        for (const auto &name : animation_names) asset.animations.emplace_back(fastgltf::Animation{.channels = {}, .samplers = {}, .name = ToFgStr(name)});
        const auto push_channel = [&](size_t animation, fastgltf::AnimationChannel target, const AnimationChannel &channel) {
            const auto &times = channel.Times;
            const uint32_t t_offset = AppendAligned<float>(bin, std::span<const float>{times});
            const uint32_t t_bv = AddBufferView(t_offset, times.size() * sizeof(float));
            const auto [t_min, t_max] = std::minmax_element(times.begin(), times.end());
            const uint32_t t_acc = AddAccessor(t_bv, times.size(), fastgltf::AccessorType::Scalar, fastgltf::ComponentType::Float, MakeBounds({double(*t_min)}), MakeBounds({double(*t_max)}));
            // Weights channels pack every target into scalar elements, and bool channels are unsigned bytes.
            const uint32_t components = channel.Target.Count;
            const bool packed = channel.Target.Component == state::Key<MorphWeightRange>();
            const auto v_type = packed || components == 1 ? fastgltf::AccessorType::Scalar : components == 2 ? fastgltf::AccessorType::Vec2 :
                components == 3                                                                              ? fastgltf::AccessorType::Vec3 :
                                                                                                               fastgltf::AccessorType::Vec4;
            const uint32_t v_count = packed ? channel.Values.size() : channel.Values.size() / components;
            uint32_t v_acc;
            if (channel.Target.Kind == ValueKind::Bool) {
                std::vector<uint8_t> flags(channel.Values.size());
                for (size_t i = 0; i < flags.size(); ++i) flags[i] = channel.Values[i] != 0.f;
                const uint32_t v_offset = AppendAligned<uint8_t>(bin, std::span<const uint8_t>{flags});
                v_acc = AddAccessor(AddBufferView(v_offset, flags.size()), v_count, v_type, fastgltf::ComponentType::UnsignedByte);
            } else {
                const uint32_t v_offset = AppendAligned<float>(bin, std::span<const float>{channel.Values});
                v_acc = AddAccessor(AddBufferView(v_offset, channel.Values.size() * sizeof(float)), v_count, v_type, fastgltf::ComponentType::Float);
            }
            auto &anim = asset.animations[animation];
            anim.samplers.emplace_back(fastgltf::AnimationSampler{.inputAccessor = t_acc, .outputAccessor = v_acc, .interpolation = FromInterp(channel.Interp)});
            target.samplerIndex = anim.samplers.size() - 1;
            anim.channels.emplace_back(std::move(target));
        };
        // A pointer channel names its target by export index.
        const auto pointer_channel = [](const gltf::PointerRow &row, size_t index, size_t element = 0) {
            return fastgltf::AnimationChannel{.samplerIndex = 0, .nodeIndex = {}, .path = fastgltf::AnimationPath::Pointer, .pointer = ToFgStr(std::vformat(row.Template, std::make_format_args(index, element)))};
        };
        for (const auto [entity, clips] : r.view<const AnimationClips>().each()) {
            for (const auto &clip : clips.Clips) {
                if (clip.Animation >= animation_names.size()) continue;
                for (const auto &source : clip.Channels) {
                    if (source.Times.empty() || source.Target.Count == 0) continue;
                    const auto *row = gltf::RowOf(source.Target);
                    if (!row) continue;
                    auto channel = source;
                    std::optional<fastgltf::AnimationChannel> target;
                    switch (row->Space) {
                        case gltf::PointerSpace::Node: {
                            std::optional<uint32_t> node;
                            if (source.Target.Component == state::Key<BoneDelta>()) {
                                // Convert the delta channel to the joint node's absolute transform.
                                const auto arm_obj = FindArmatureObject(r, entity);
                                const auto *arm = arm_obj != state::Null ? r.try_get<const Armature>(r.get<const ArmatureObject>(arm_obj).Entity) : nullptr;
                                const auto index = r.get<const BoneIndex>(entity).Index;
                                if (arm && index < arm->Bones.size()) {
                                    node = arm->Bones[index].JointNodeIndex;
                                    ConvertBoneChannel(channel, arm->Bones[index].RestLocal, false);
                                }
                            } else if (const auto it = node_index_of.find(entity); it != node_index_of.end()) {
                                node = it->second;
                            }
                            if (!node) break;
                            // Transforms and whole weights are classic channels. Visibility and one weight of a multi-weight mesh are pointer channels.
                            const bool weights = source.Target.Component == state::Key<MorphWeightRange>();
                            if (source.Target.Component == state::Key<Visibility>()) target = pointer_channel(*row, *node);
                            else if (weights && source.Target.Count == 1 && r.get<const MorphWeightRange>(entity).Weights.Count > 1) target = pointer_channel(*row, *node, source.Target.Offset / sizeof(float));
                            else target = fastgltf::AnimationChannel{.samplerIndex = 0, .nodeIndex = *node, .path = weights ? fastgltf::AnimationPath::Weights : gltf::NodePath(*row), .pointer = {}};
                            break;
                        }
                        case gltf::PointerSpace::Material:
                            if (source.Target.Index < material_indices.size() && material_indices[source.Target.Index]) target = pointer_channel(*row, *material_indices[source.Target.Index]);
                            break;
                        case gltf::PointerSpace::Camera:
                            if (const auto it = camera_entity_to_index.find(entity); it != camera_entity_to_index.end()) target = pointer_channel(*row, it->second);
                            break;
                        case gltf::PointerSpace::Light:
                            if (const auto it = light_entity_to_index.find(entity); it != light_entity_to_index.end()) target = pointer_channel(*row, it->second);
                            break;
                        case gltf::PointerSpace::ImageLight:
                            // The one exported image-based light is at index 0.
                            if (src_assets && src_assets->ImageBasedLight) target = pointer_channel(*row, 0);
                            break;
                    }
                    if (!target) continue;
                    uses_animation_pointer |= target->path == fastgltf::AnimationPath::Pointer;
                    push_channel(clip.Animation, std::move(*target), channel);
                }
            }
        }
        std::erase_if(asset.animations, [](const auto &animation) { return animation.channels.empty(); });
    }

    asset.meshes.reserve(mesh_groups.size());
    const auto &arenas = meshes.Arenas();
    const auto emit_non_triangle_attrs = [&](fastgltf::pmr::SmallVector<fastgltf::Attribute, 4> &out, uint32_t store_id) {
        const auto &record = meshes.Get(store_id);
        if (const auto point_normals = arenas.PointNormals.Get(record.PointNormals); !point_normals.empty()) {
            out.emplace_back(fastgltf::Attribute{"NORMAL", AddDataAccessor(point_normals, fastgltf::AccessorType::Vec3, fastgltf::ComponentType::Float, fastgltf::BufferTarget::ArrayBuffer)});
        }
        if (const auto colors = arenas.CornerColors.Get(record.CornerColors); !colors.empty()) {
            out.emplace_back(fastgltf::Attribute{"COLOR_0", AddDataAccessor(colors, fastgltf::AccessorType::Vec4, fastgltf::ComponentType::Float, fastgltf::BufferTarget::ArrayBuffer)});
        }
    };
    for (uint32_t mi = 0; mi < mesh_groups.size(); ++mi) {
        const auto &group = mesh_groups[mi];
        fastgltf::pmr::MaybeSmallVector<fastgltf::Primitive, 2> primitives;
        fastgltf::pmr::MaybeSmallVector<fastgltf::num> default_weights;
        // Non-triangle primitives omit targets, materials, and mappings.
        const auto push_prim = [&](fastgltf::PrimitiveType type, auto &&attrs, const fastgltf::Optional<size_t> &indices = {}) {
            primitives.emplace_back(fastgltf::Primitive{
                .attributes = std::forward<decltype(attrs)>(attrs),
                .type = type,
                .targets = {},
                .indicesAccessor = indices,
                .materialIndex = {},
                .mappings = {},
                .dracoCompression = nullptr,
            });
        };

        // Each fan corner pairs its mesh vertex index with its index into the corner-domain arenas.
        // Every distinct corner tuple over the emitted channels becomes one export vertex, so vertices split exactly where corner attributes diverge.
        if (group.Triangles != state::Null) {
            const auto &mesh = GetMesh(r, group.Triangles);
            const auto store_id = mesh.GetStoreId();
            const auto &record = meshes.Get(store_id);
            const auto vertices = arenas.Vertices.Get(record.Vertices);
            const auto total_vcount = vertices.size();
            const auto face_primitives = arenas.ElementPrimitives.Get(record.ElementPrimitives);
            const auto primitive_materials = arenas.PrimitiveMaterials.Get(record.PrimitiveMaterials);
            const auto corner_normals = meshes.GetCornerNormals(mesh);
            const auto corner_tangents = arenas.CornerTangents.Get(record.CornerTangents);
            const auto corner_colors = arenas.CornerColors.Get(record.CornerColors);
            const std::array corner_uv_sets{arenas.CornerUvs.Get(record.CornerUvs[0]), arenas.CornerUvs.Get(record.CornerUvs[1]), arenas.CornerUvs.Get(record.CornerUvs[2]), arenas.CornerUvs.Get(record.CornerUvs[3])};
            // Derive one primitive layout for runtime-created meshes.
            const auto *layout_ptr = r.try_get<const MeshSourceLayout>(group.Triangles);
            const MeshSourceLayout synthesized_layout = layout_ptr ? MeshSourceLayout{} : [&] {
                MeshSourceLayout out;
                // Triangle-mesh normals are always derivable, so runtime meshes always emit them.
                uint32_t flags = MeshAttributeBit_Normal;
                if (!corner_tangents.empty()) flags |= MeshAttributeBit_Tangent;
                if (!corner_colors.empty()) flags |= MeshAttributeBit_Color0;
                for (uint32_t set = 0; set < corner_uv_sets.size(); ++set) {
                    if (!corner_uv_sets[set].empty()) flags |= MeshAttributeBit_TexCoord0 << set;
                }
                out.AttributeFlags = {flags};
                out.HasSourceIndices = {1};
                out.Colors0ComponentCount = 4;
                return out;
            }();
            const auto &layout = layout_ptr ? *layout_ptr : synthesized_layout;
            const auto prim_count = layout.AttributeFlags.size();

            // Gather primitive corners in fan-triangulation order.
            struct CornerRef {
                uint32_t Vertex, Corner;
            };
            const auto face_first_tris = arenas.FaceFirstTriangles.Get(record.FaceData);
            std::vector<std::vector<CornerRef>> corners_per_prim(prim_count);
            uint32_t fi = 0;
            for (const auto fh : mesh.faces()) {
                const uint32_t p = fi < face_primitives.size() ? face_primitives[fi] : 0u;
                if (p < prim_count) {
                    std::array<uint32_t, 16> fv{};
                    uint32_t fv_count = 0;
                    for (const auto vh : mesh.fv_range(fh)) {
                        if (fv_count < fv.size()) fv[fv_count] = *vh;
                        ++fv_count;
                    }
                    if (fv_count >= 3) {
                        auto &out = corners_per_prim[p];
                        const auto corner_base = face_first_tris[fi] * 3;
                        for (uint32_t k = 1; k + 1 < fv_count; ++k) {
                            out.emplace_back(fv[0], corner_base + (k - 1) * 3);
                            out.emplace_back(fv[k], corner_base + (k - 1) * 3 + 1);
                            out.emplace_back(fv[k + 1], corner_base + (k - 1) * 3 + 2);
                        }
                    }
                }
                ++fi;
            }

            // Skin / morph spans (empty when the mesh lacks the channel).
            const auto bd_span = arenas.BoneDeform.Get(record.BoneDeform);
            const bool has_skin = bd_span.size() == total_vcount && total_vcount > 0;
            const uint32_t target_count = (total_vcount > 0) ? record.MorphTargetCount : 0u;
            const auto mt_span = arenas.MorphTargets.Get(record.MorphTargets);
            // CreateMesh writes 0 when source lacked normal deltas, so any non-zero means source had them.
            const bool has_normal_deltas = std::ranges::any_of(mt_span, [](const auto &m) { return m.NormalDelta != vec3{0}; });
            const bool has_tangent_deltas = !layout.MorphTangentDeltas.empty();

            // Emit channels present in at least one primitive.
            uint32_t any_flags = 0;
            for (const auto f : layout.AttributeFlags) any_flags |= f;
            const bool have_flags = layout.AttributeFlags.size() == prim_count;

            for (uint32_t prim_idx = 0; prim_idx < prim_count; ++prim_idx) {
                auto &prim_corners = corners_per_prim[prim_idx];
                if (prim_corners.empty()) continue;
                const auto flags = have_flags ? layout.AttributeFlags[prim_idx] : ~0u;

                const bool emit_normal = (any_flags & MeshAttributeBit_Normal) && (flags & MeshAttributeBit_Normal) && !corner_normals.empty();
                const bool emit_tangent = (any_flags & MeshAttributeBit_Tangent) && (flags & MeshAttributeBit_Tangent) && !corner_tangents.empty();
                const bool emit_color = (any_flags & MeshAttributeBit_Color0) && (flags & MeshAttributeBit_Color0) && !corner_colors.empty();
                std::array<bool, 4> emit_uv{};
                for (uint32_t set = 0; set < emit_uv.size(); ++set) {
                    emit_uv[set] = (any_flags & (MeshAttributeBit_TexCoord0 << set)) && (flags & (MeshAttributeBit_TexCoord0 << set)) && !corner_uv_sets[set].empty();
                }

                // Indexed primitives merge identical corner tuples into unique export vertices in first-seen order.
                // Non-indexed sources emit the corner stream directly.
                const bool emit_indices = prim_idx < layout.HasSourceIndices.size() ? layout.HasSourceIndices[prim_idx] != 0 : true;
                std::vector<CornerRef> export_refs;
                std::vector<uint32_t> indices;
                if (emit_indices) {
                    uint32_t stride = sizeof(uint32_t) + (emit_normal ? sizeof(vec3) : 0u) + (emit_tangent ? sizeof(vec4) : 0u) + (emit_color ? sizeof(vec4) : 0u);
                    for (const auto uv : emit_uv) stride += uv ? sizeof(vec2) : 0u;
                    std::vector<std::byte> keys(prim_corners.size() * size_t(stride));
                    for (size_t i = 0; i < prim_corners.size(); ++i) {
                        auto *dst = keys.data() + i * stride;
                        const auto append = [&dst](const auto &v) {
                            std::memcpy(dst, &v, sizeof(v));
                            dst += sizeof(v);
                        };
                        const auto &c = prim_corners[i];
                        append(c.Vertex);
                        if (emit_normal) append(corner_normals[c.Corner]);
                        if (emit_tangent) append(corner_tangents[c.Corner]);
                        if (emit_color) append(corner_colors[c.Corner]);
                        for (uint32_t set = 0; set < emit_uv.size(); ++set) {
                            if (emit_uv[set]) append(corner_uv_sets[set][c.Corner]);
                        }
                    }
                    export_refs.reserve(prim_corners.size());
                    indices.reserve(prim_corners.size());
                    std::unordered_map<std::string_view, uint32_t> dot_index;
                    dot_index.reserve(prim_corners.size());
                    for (size_t i = 0; i < prim_corners.size(); ++i) {
                        const std::string_view key{reinterpret_cast<const char *>(keys.data() + i * stride), stride};
                        const auto [it, inserted] = dot_index.try_emplace(key, uint32_t(export_refs.size()));
                        if (inserted) export_refs.emplace_back(prim_corners[i]);
                        indices.emplace_back(it->second);
                    }
                } else {
                    export_refs = std::move(prim_corners);
                }
                const uint32_t export_count = export_refs.size();

                const auto gather_corner = [&]<typename T>(std::span<const T> src) {
                    std::vector<T> out(export_count);
                    for (uint32_t i = 0; i < export_count; ++i) out[i] = src[export_refs[i].Corner];
                    return out;
                };

                std::vector<vec3> positions(export_count);
                for (uint32_t i = 0; i < export_count; ++i) positions[i] = vertices[export_refs[i].Vertex].Position;
                fastgltf::pmr::SmallVector<fastgltf::Attribute, 4> prim_attrs;
                prim_attrs.emplace_back(fastgltf::Attribute{"POSITION", AddVec3Accessor(positions, true, fastgltf::BufferTarget::ArrayBuffer)});

                if (emit_normal) {
                    const auto normals = gather_corner(corner_normals);
                    prim_attrs.emplace_back(fastgltf::Attribute{"NORMAL", AddVec3Accessor(normals, false, fastgltf::BufferTarget::ArrayBuffer)});
                }
                if (emit_tangent) {
                    const auto tangents = gather_corner(corner_tangents);
                    prim_attrs.emplace_back(fastgltf::Attribute{"TANGENT", AddDataAccessor(std::span<const vec4>(tangents), fastgltf::AccessorType::Vec4, fastgltf::ComponentType::Float, fastgltf::BufferTarget::ArrayBuffer)});
                }
                if (emit_color) {
                    const auto colors = gather_corner(corner_colors);
                    EmitColor0Values(prim_attrs, colors, layout.Colors0ComponentCount);
                }
                static constexpr std::array UvNames{"TEXCOORD_0", "TEXCOORD_1", "TEXCOORD_2", "TEXCOORD_3"};
                for (uint32_t set = 0; set < emit_uv.size(); ++set) {
                    if (!emit_uv[set]) continue;
                    const auto uvs = gather_corner(corner_uv_sets[set]);
                    prim_attrs.emplace_back(fastgltf::Attribute{UvNames[set], AddDataAccessor(std::span<const vec2>(uvs), fastgltf::AccessorType::Vec2, fastgltf::ComponentType::Float, fastgltf::BufferTarget::ArrayBuffer)});
                }

                if (has_skin) {
                    // Convert uvec4 joints to uint16_t[4] strided directly into bin.
                    const uint32_t j_off = bin.size();
                    bin.resize(j_off + export_count * 4 * sizeof(uint16_t));
                    auto *jp = reinterpret_cast<uint16_t *>(bin.data() + j_off);
                    for (uint32_t i = 0; i < export_count; ++i) {
                        const auto &j = bd_span[export_refs[i].Vertex].Joints;
                        jp[i * 4 + 0] = uint16_t(j.x);
                        jp[i * 4 + 1] = uint16_t(j.y);
                        jp[i * 4 + 2] = uint16_t(j.z);
                        jp[i * 4 + 3] = uint16_t(j.w);
                    }
                    while (bin.size() % 4 != 0) bin.emplace_back(std::byte{0});
                    const uint32_t j_bv = AddBufferView(j_off, export_count * 4 * sizeof(uint16_t), {}, fastgltf::BufferTarget::ArrayBuffer);
                    prim_attrs.emplace_back(fastgltf::Attribute{"JOINTS_0", AddAccessor(j_bv, export_count, fastgltf::AccessorType::Vec4, fastgltf::ComponentType::UnsignedShort)});
                    std::vector<vec4> weights(export_count);
                    for (uint32_t i = 0; i < export_count; ++i) weights[i] = bd_span[export_refs[i].Vertex].Weights;
                    prim_attrs.emplace_back(fastgltf::Attribute{"WEIGHTS_0", AddDataAccessor(std::span<const vec4>(weights), fastgltf::AccessorType::Vec4, fastgltf::ComponentType::Float, fastgltf::BufferTarget::ArrayBuffer)});
                }

                std::pmr::vector<fastgltf::pmr::SmallVector<fastgltf::Attribute, 4>> prim_targets;
                if (target_count > 0) {
                    prim_targets.reserve(target_count);
                    std::vector<vec3> deltas(export_count);
                    for (uint32_t t = 0; t < target_count; ++t) {
                        fastgltf::pmr::SmallVector<fastgltf::Attribute, 4> tattrs;
                        const auto target_base = t * total_vcount;
                        for (uint32_t i = 0; i < export_count; ++i) deltas[i] = mt_span[target_base + export_refs[i].Vertex].PositionDelta;
                        tattrs.emplace_back(fastgltf::Attribute{"POSITION", AddVec3Accessor(deltas, false, fastgltf::BufferTarget::ArrayBuffer)});
                        if (has_normal_deltas) {
                            for (uint32_t i = 0; i < export_count; ++i) deltas[i] = mt_span[target_base + export_refs[i].Vertex].NormalDelta;
                            tattrs.emplace_back(fastgltf::Attribute{"NORMAL", AddVec3Accessor(deltas, false, fastgltf::BufferTarget::ArrayBuffer)});
                        }
                        if (has_tangent_deltas) {
                            for (uint32_t i = 0; i < export_count; ++i) deltas[i] = layout.MorphTangentDeltas[target_base + export_refs[i].Vertex];
                            tattrs.emplace_back(fastgltf::Attribute{"TANGENT", AddVec3Accessor(deltas, false, fastgltf::BufferTarget::ArrayBuffer)});
                        }
                        prim_targets.emplace_back(std::move(tattrs));
                    }
                }

                fastgltf::Optional<size_t> indices_accessor;
                if (emit_indices) {
                    indices_accessor = AddDataAccessor(std::span<const uint32_t>(indices), fastgltf::AccessorType::Scalar, fastgltf::ComponentType::UnsignedInt, fastgltf::BufferTarget::ElementArrayBuffer);
                }

                fastgltf::Optional<size_t> material_index;
                if (prim_idx < primitive_materials.size()) {
                    const auto reg_idx = primitive_materials[prim_idx];
                    if (reg_idx < material_indices.size()) material_index = material_indices[reg_idx];
                }

                std::vector<fastgltf::Optional<size_t>> mappings;
                if (prim_idx < layout.VariantMappings.size()) {
                    for (const auto &m : layout.VariantMappings[prim_idx]) {
                        mappings.emplace_back(m && *m < material_indices.size() ? material_indices[*m] : fastgltf::Optional<size_t>{});
                    }
                }

                primitives.emplace_back(fastgltf::Primitive{
                    .attributes = std::move(prim_attrs),
                    .type = fastgltf::PrimitiveType::Triangles,
                    .targets = std::move(prim_targets),
                    .indicesAccessor = indices_accessor,
                    .materialIndex = material_index,
                    .mappings = std::move(mappings),
                    .dracoCompression = nullptr,
                });
            }

            const auto &dw = meshes.Get(store_id).DefaultMorphWeights;
            if (!dw.empty()) {
                default_weights.reserve(dw.size());
                for (const auto w : dw) default_weights.emplace_back(w);
            }
        }

        if (group.Lines != state::Null) {
            const auto &mesh = GetMesh(r, group.Lines);
            const auto vertices = mesh.GetVerticesSpan();
            if (!vertices.empty() && mesh.EdgeCount() > 0) {
                std::vector<uint32_t> idx;
                idx.reserve(mesh.EdgeCount() * 2);
                for (const auto eh : mesh.edges()) {
                    const auto h0 = mesh.GetHalfedge(eh, 0);
                    idx.emplace_back(*mesh.GetFromVertex(h0));
                    idx.emplace_back(*mesh.GetToVertex(h0));
                }
                fastgltf::pmr::SmallVector<fastgltf::Attribute, 4> attrs;
                attrs.emplace_back(fastgltf::Attribute{"POSITION", AddPositionFieldAccessor.template operator()<Vertex>(vertices, &Vertex::Position, fastgltf::BufferTarget::ArrayBuffer)});
                emit_non_triangle_attrs(attrs, mesh.GetStoreId());
                push_prim(fastgltf::PrimitiveType::Lines, std::move(attrs), AddDataAccessor(std::span<const uint32_t>(idx), fastgltf::AccessorType::Scalar, fastgltf::ComponentType::UnsignedInt, fastgltf::BufferTarget::ElementArrayBuffer));
            }
        }

        if (group.Points != state::Null) {
            const auto &mesh = GetMesh(r, group.Points);
            const auto vertices = mesh.GetVerticesSpan();
            if (!vertices.empty()) {
                fastgltf::pmr::SmallVector<fastgltf::Attribute, 4> attrs;
                attrs.emplace_back(fastgltf::Attribute{"POSITION", AddPositionFieldAccessor.template operator()<Vertex>(vertices, &Vertex::Position, fastgltf::BufferTarget::ArrayBuffer)});
                emit_non_triangle_attrs(attrs, mesh.GetStoreId());
                push_prim(fastgltf::PrimitiveType::Points, std::move(attrs));
            }
        }

        // glTF requires >= 1 primitive per mesh; emit a single degenerate point.
        if (primitives.empty()) {
            const vec3 stub{0.f, 0.f, 0.f};
            const uint32_t pos_acc = AddVec3Accessor(std::span<const vec3>(&stub, 1), true, fastgltf::BufferTarget::ArrayBuffer);
            fastgltf::pmr::SmallVector<fastgltf::Attribute, 4> attrs;
            attrs.emplace_back(fastgltf::Attribute{"POSITION", pos_acc});
            push_prim(fastgltf::PrimitiveType::Points, std::move(attrs));
        }

        asset.meshes.emplace_back(fastgltf::Mesh{.primitives = std::move(primitives), .weights = std::move(default_weights), .name = ToFgStr(group.Name)});
    }

    // Emit one fastgltf::Skin per imported skin, in ascending source-skin-index order so the skins array is stable regardless of armature grouping.
    // Rest locals and bone names live in the engine's Armature and round-trip through node entries instead.
    std::unordered_map<uint32_t, uint32_t> skin_remap;
    {
        std::vector<const ArmatureImportedSkin *> all_skins;
        for (const auto [_, arm] : r.view<const Armature>().each()) {
            for (const auto &skin : arm.Skins) all_skins.emplace_back(&skin);
        }
        std::ranges::sort(all_skins, {}, [](const auto *skin) { return skin->SkinIndex; });
        for (const auto *skin : all_skins) {
            fastgltf::pmr::MaybeSmallVector<size_t> joints;
            joints.reserve(skin->OrderedJointNodeIndices.size());
            for (const auto j : skin->OrderedJointNodeIndices) joints.emplace_back(j);

            const auto ibm = [&]() -> fastgltf::Optional<size_t> {
                if (skin->InverseBindMatrices.empty()) return {};
                return AddDataAccessor(std::span<const mat4>(skin->InverseBindMatrices), fastgltf::AccessorType::Mat4, fastgltf::ComponentType::Float);
            }();

            skin_remap[skin->SkinIndex] = asset.skins.size();
            asset.skins.emplace_back(fastgltf::Skin{
                .inverseBindMatrices = ibm,
                .skeleton = ToFgOpt<size_t>(skin->SkeletonNodeIndex),
                .joints = std::move(joints),
                .name = ToFgStr(skin->Name),
            });
        }
    }

    // Cameras / lights: emit in the order gathered above.
    asset.cameras.reserve(camera_entities_ordered.size());
    for (const auto entity : camera_entities_ordered) {
        const auto *node = r.try_get<const GltfNode>(entity);
        auto lens = *LensOf(r, entity);
        std::visit([&](auto &l) { OverlayRest(r, entity, state::Key<std::remove_reference_t<decltype(l)>>(), 0, std::as_writable_bytes(std::span{&l, 1})); }, lens);
        asset.cameras.emplace_back(ConvertCameraToFg(lens, node ? node->CameraName : std::string_view{}));
    }
    asset.lights.reserve(light_entities_ordered.size());
    for (const auto entity : light_entities_ordered) {
        const auto *node = r.try_get<const GltfNode>(entity);
        auto pl = r.get<const PunctualLight>(entity);
        OverlayRest(r, entity, state::Key<PunctualLight>(), 0, std::as_writable_bytes(std::span{&pl, 1}));
        asset.lights.emplace_back(ConvertLightToFg(pl, node ? node->LightName : std::string_view{}));
    }

    // KHR_implicit_shapes: dedupe primitive shapes referenced by colliders/triggers into asset.shapes.
    using ShapeKey = std::tuple<uint8_t, float, float, float, float>;
    const auto to_fg_shape = [](const PhysicsShape &s) -> std::optional<fastgltf::Shape> {
        return std::visit(
            overloaded{
                [](const physics::Box &b) -> std::optional<fastgltf::Shape> {
                    return fastgltf::BoxShape{.size = std::bit_cast<fastgltf::math::fvec3>(b.Size)};
                },
                [](const physics::Sphere &s) -> std::optional<fastgltf::Shape> {
                    return fastgltf::SphereShape{.radius = s.Radius};
                },
                [](const physics::Capsule &c) -> std::optional<fastgltf::Shape> {
                    return fastgltf::CapsuleShape{.height = c.Height, .radiusBottom = c.RadiusBottom, .radiusTop = c.RadiusTop};
                },
                [](const physics::Cylinder &c) -> std::optional<fastgltf::Shape> {
                    return fastgltf::CylinderShape{.height = c.Height, .radiusBottom = c.RadiusBottom, .radiusTop = c.RadiusTop};
                },
                [](const physics::Plane &p) -> std::optional<fastgltf::Shape> {
                    return fastgltf::PlaneShape{.sizeX = p.SizeX, .sizeZ = p.SizeZ, .doubleSided = p.DoubleSided};
                },
                [](const auto &) -> std::optional<fastgltf::Shape> { return std::nullopt; },
            },
            s
        );
    };
    const auto shape_key = [](const fastgltf::Shape &s) -> ShapeKey {
        return std::visit(
            overloaded{
                [](const fastgltf::BoxShape &b) { return ShapeKey{0, b.size[0], b.size[1], b.size[2], 0}; },
                [](const fastgltf::SphereShape &s) { return ShapeKey{1, float(s.radius), 0, 0, 0}; },
                [](const fastgltf::CapsuleShape &c) { return ShapeKey{2, float(c.height), float(c.radiusBottom), float(c.radiusTop), 0}; },
                [](const fastgltf::CylinderShape &c) { return ShapeKey{3, float(c.height), float(c.radiusBottom), float(c.radiusTop), 0}; },
                [](const fastgltf::PlaneShape &p) { return ShapeKey{4, float(p.sizeX), float(p.sizeZ), p.doubleSided ? 1.f : 0.f, 0}; },
            },
            s
        );
    };
    std::map<ShapeKey, size_t> shape_index_by_key;
    const auto emit_shape_index = [&](const fastgltf::Shape &s) -> size_t {
        const auto key = shape_key(s);
        if (auto it = shape_index_by_key.find(key); it != shape_index_by_key.end()) return it->second;
        asset.shapes.emplace_back(s);
        const auto idx = asset.shapes.size() - 1;
        shape_index_by_key[key] = idx;
        return idx;
    };
    using TriggerVariant = std::variant<fastgltf::GeometryTrigger, fastgltf::NodeTrigger>;
    // Populates `rb.collider` or `rb.trigger` from the owner's ColliderShape + (Trigger|Collider)Material.
    // Shared between the owner-no-offset path and the synthetic-offset-child path.
    const auto populate_collider_extension = [&](state::Entity owner, fastgltf::PhysicsRigidBody &rb) {
        const auto *cs = r.try_get<const ColliderShape>(owner);
        if (!cs) return;
        const bool is_trigger = r.all_of<TriggerTag>(owner);
        fastgltf::Optional<size_t> collider_mesh_idx;
        if (cs->MeshEntity != state::Null) {
            if (const auto mit = mesh_entity_to_index.find(cs->MeshEntity); mit != mesh_entity_to_index.end()) collider_mesh_idx = mit->second;
        }
        if (!is_trigger) {
            fastgltf::Collider collider{};
            if (auto fg_shape = to_fg_shape(cs->Shape)) {
                collider.geometry.shape = emit_shape_index(*fg_shape);
            } else if (std::holds_alternative<physics::ConvexHull>(cs->Shape)) {
                if (collider_mesh_idx) collider.geometry.mesh = *collider_mesh_idx;
                collider.geometry.convexHull = true;
            } else if (std::holds_alternative<physics::TriangleMesh>(cs->Shape)) {
                if (collider_mesh_idx) collider.geometry.mesh = *collider_mesh_idx;
            }
            if (const auto *cm = r.try_get<const ColliderMaterial>(owner)) {
                if (const auto mit = physics_material_to_index.find(cm->PhysicsMaterialEntity); mit != physics_material_to_index.end()) collider.physicsMaterial = mit->second;
                if (const auto fit = collision_filter_to_index.find(cm->CollisionFilterEntity); fit != collision_filter_to_index.end()) collider.collisionFilter = fit->second;
            }
            rb.collider = fastgltf::Optional<fastgltf::Collider>{std::move(collider)};
        } else {
            fastgltf::GeometryTrigger gt{};
            if (auto fg_shape = to_fg_shape(cs->Shape)) {
                gt.geometry.shape = emit_shape_index(*fg_shape);
            } else if (collider_mesh_idx) {
                gt.geometry.mesh = *collider_mesh_idx;
                gt.geometry.convexHull = std::holds_alternative<physics::ConvexHull>(cs->Shape);
            }
            if (const auto *cm = r.try_get<const ColliderMaterial>(owner)) {
                if (const auto fit = collision_filter_to_index.find(cm->CollisionFilterEntity); fit != collision_filter_to_index.end()) gt.collisionFilter = fit->second;
            }
            rb.trigger = fastgltf::Optional<TriggerVariant>{TriggerVariant{std::move(gt)}};
        }
    };

    // A bone Transform contains the live pose, while its joint node TRS contains the rest pose.
    // The rest lives in the Armature, so map each bone to it. (Non-bone nodes keep their authored local in Transform.)
    std::unordered_map<state::Entity, Transform> bone_rest;
    for (const auto [ao_entity, ao] : r.view<const ArmatureObject>().each()) {
        const auto &arm = r.get<const Armature>(ao.Entity);
        for (uint32_t i = 0; i < ao.BoneEntities.size() && i < arm.Bones.size(); ++i) {
            if (ao.BoneEntities[i] != state::Null) bone_rest.emplace(ao.BoneEntities[i], arm.Bones[i].RestLocal);
        }
    }

    // Use bone rest poses and authored object transforms for frame-independent export.
    std::unordered_map<state::Entity, mat4> rest_world;
    const auto rest_world_of = [&](this const auto &self, state::Entity e) -> mat4 {
        if (const auto it = rest_world.find(e); it != rest_world.end()) return it->second;
        const auto bit = bone_rest.find(e);
        const mat4 local = ToMatrix(bit != bone_rest.end() ? bit->second : r.get<const Transform>(e));
        const auto *node = r.try_get<const SceneNode>(e);
        const mat4 world = node && node->Parent != state::Null ? self(node->Parent) * local : local;
        rest_world.emplace(e, world);
        return world;
    };

    // Emit nodes directly from registry components while retaining gaps in the source node indices.
    asset.nodes.reserve(nodes.size());
    bool uses_gpu_instancing = false;
    bool uses_physics_rigid_bodies = false;
    // KHR_audio_rigid_bodies acoustic materials and surfaces, deduped by value across node instances.
    std::vector<AcousticMaterial> audio_rigid_body_materials;
    std::vector<ContactSurface> audio_rigid_body_surfaces;
    // Returns a deduplicated resource index and preserves emitted order in `mirror`.
    const auto dedupe = [](auto &mirror, const auto *value, auto &&emit) -> fastgltf::Optional<size_t> {
        if (!value) return {};
        const auto it = std::ranges::find(mirror, *value);
        if (it != mirror.end()) return size_t(std::distance(mirror.begin(), it));
        const auto index = emit();
        mirror.emplace_back(*value);
        return index;
    };
    for (uint32_t ni = 0; ni < nodes.size(); ++ni) {
        const auto &export_node = nodes[ni];
        const auto entity = export_node.Entity;

        fastgltf::pmr::MaybeSmallVector<size_t> children;
        children.reserve(export_node.Children.size());
        for (const auto child : export_node.Children) children.emplace_back(child);

        if (export_node.OffsetOwner != state::Null) {
            const auto owner = export_node.OffsetOwner;
            const auto &cs = r.get<const ColliderShape>(owner);
            auto rb = std::make_unique<fastgltf::PhysicsRigidBody>();
            populate_collider_extension(owner, *rb);
            uses_physics_rigid_bodies = true;
            fastgltf::Node node{};
            node.children = std::move(children);
            node.transform = fastgltf::TRS{.translation = std::bit_cast<fastgltf::math::fvec3>(cs.LocalOffset)};
            node.physicsRigidBody = std::move(rb);
            asset.nodes.emplace_back(std::move(node));
            continue;
        }

        fastgltf::Optional<size_t> mesh_index, camera_index, light_index, skin_index;
        if (const auto *inst = r.try_get<const Instance>(entity)) {
            if (const auto it = mesh_entity_to_index.find(inst->Entity); it != mesh_entity_to_index.end()) mesh_index = it->second;
        }
        if (const auto cit = camera_entity_to_index.find(entity); cit != camera_entity_to_index.end()) camera_index = cit->second;
        if (const auto lit = light_entity_to_index.find(entity); lit != light_entity_to_index.end()) light_index = lit->second;
        if (const auto *am = r.try_get<const ArmatureModifier>(entity)) {
            if (const auto *arm = r.try_get<const Armature>(am->ArmatureEntity); arm && am->SkinSlot < arm->Skins.size()) {
                if (const auto sit = skin_remap.find(arm->Skins[am->SkinSlot].SkinIndex); sit != skin_remap.end()) skin_index = sit->second;
            }
        }

        const auto *gltf_node = r.try_get<const GltfNode>(entity);
        std::string node_name;
        if (!gltf_node || !gltf_node->EmptyName) {
            if (gltf_node && !gltf_node->Name.empty()) node_name = gltf_node->Name;
            else if (const auto *nm = r.try_get<const Name>(entity)) node_name = nm->Value;
        }

        const auto &world_transform = r.get<const WorldTransform>(entity);

        // The source parent restores non-joint ancestors removed from the runtime bone hierarchy.
        // Derive local transforms from rest-world transforms when source and runtime parents differ.
        const Transform local_transform = [&] {
            if (r.all_of<ArmatureModifier>(entity)) return Transform{}; // Skinned mesh node transform is spec-ignored.
            const auto *node = r.try_get<const SceneNode>(entity);
            if (export_node.Parent) {
                const auto src_parent = nodes[*export_node.Parent].Entity;
                if (src_parent != state::Null && (!node || node->Parent != src_parent)) {
                    return ToTransform(numeric::Inverse(rest_world_of(src_parent)) * rest_world_of(entity));
                }
            }
            if (const auto it = bone_rest.find(entity); it != bone_rest.end()) return it->second;
            return r.get<const Transform>(entity);
        }();

        // EXT_mesh_gpu_instancing. Per-instance TRS = node.WorldTransform^-1 * instance.WorldTransform.
        // Emit only channels that aren't uniformly default.
        // spec requires >=1 attribute so fall back to TRANSLATION when everything's default.
        std::vector<Transform> instance_worlds;
        instance_worlds.reserve(export_node.Instances.size());
        for (const auto instance : export_node.Instances) instance_worlds.emplace_back(r.get<const WorldTransform>(instance));
        const bool needs_instancing = mesh_index.has_value() && instance_worlds.size() > 1;
        if (needs_instancing) uses_gpu_instancing = true;
        std::pmr::vector<fastgltf::Attribute> instancing;
        if (needs_instancing) {
            const uint32_t count = instance_worlds.size();
            const mat4 node_world_inv = numeric::Inverse(ToMatrix(world_transform));

            std::vector<vec3> translations(count);
            std::vector<vec4> rotations(count); // xyzw
            std::vector<vec3> scales(count);
            bool any_t = false, any_r = false, any_s = false;
            for (uint32_t i = 0; i < count; ++i) {
                const Transform local = ToTransform(node_world_inv * ToMatrix(instance_worlds[i]));
                translations[i] = local.P;
                rotations[i] = std::bit_cast<vec4>(local.R);
                scales[i] = local.S;
                if (local.P != vec3{0.f}) any_t = true;
                if (local.R != quat{1, 0, 0, 0}) any_r = true;
                if (local.S != vec3{1.f}) any_s = true;
            }

            const auto add_vec3 = [&](const char *name, std::span<const vec3> data) {
                instancing.emplace_back(fastgltf::Attribute{name, AddDataAccessor(data, fastgltf::AccessorType::Vec3, fastgltf::ComponentType::Float)});
            };
            if (any_t) add_vec3("TRANSLATION", translations);
            if (any_r) {
                const uint32_t acc = AddDataAccessor(std::span<const vec4>(rotations), fastgltf::AccessorType::Vec4, fastgltf::ComponentType::Float);
                instancing.emplace_back(fastgltf::Attribute{"ROTATION", acc});
            }
            if (any_s) add_vec3("SCALE", scales);
            if (!any_t && !any_r && !any_s) add_vec3("TRANSLATION", translations);
        }

        const auto *motion = r.try_get<const PhysicsMotion>(entity);
        const auto *velocity = r.try_get<const PhysicsVelocity>(entity);
        const auto *cs = r.try_get<const ColliderShape>(entity);
        const auto *tn = r.try_get<const TriggerNodes>(entity);
        const auto *pj = r.try_get<const PhysicsJoint>(entity);
        const bool is_trigger = r.all_of<TriggerTag>(entity);
        // An offset collider emits its geometry on the synthetic child, so the owner node carries none.
        const bool cs_on_owner = cs && !export_node.OffsetChild;
        fastgltf::Optional<size_t> collider_mesh_idx;
        if (cs_on_owner && cs->MeshEntity != state::Null) {
            if (const auto mit = mesh_entity_to_index.find(cs->MeshEntity); mit != mesh_entity_to_index.end()) collider_mesh_idx = mit->second;
        }

        std::unique_ptr<fastgltf::PhysicsRigidBody> physics_rigid_body;
        if (motion || cs_on_owner || tn || pj) {
            physics_rigid_body = std::make_unique<fastgltf::PhysicsRigidBody>();
            uses_physics_rigid_bodies = true;

            if (motion) {
                fastgltf::Motion fg_motion{};
                fg_motion.isKinematic = motion->IsKinematic;
                if (motion->Mass) fg_motion.mass = fastgltf::Optional<fastgltf::num>{*motion->Mass};
                if (motion->CenterOfMass) fg_motion.centerOfMass = std::bit_cast<fastgltf::math::fvec3>(*motion->CenterOfMass);
                if (motion->InertiaDiagonal) {
                    fg_motion.inertialDiagonal = fastgltf::Optional<fastgltf::math::fvec3>{std::bit_cast<fastgltf::math::fvec3>(*motion->InertiaDiagonal)};
                }
                if (motion->InertiaOrientation) {
                    fg_motion.inertialOrientation = fastgltf::Optional<fastgltf::math::fvec4>{std::bit_cast<fastgltf::math::fvec4>(*motion->InertiaOrientation)};
                }
                fg_motion.gravityFactor = motion->GravityFactor;
                if (velocity) {
                    fg_motion.linearVelocity = std::bit_cast<fastgltf::math::fvec3>(velocity->Linear);
                    fg_motion.angularVelocity = std::bit_cast<fastgltf::math::fvec3>(velocity->Angular);
                }
                physics_rigid_body->motion = fastgltf::Optional<fastgltf::Motion>{std::move(fg_motion)};
            }

            if (cs_on_owner) populate_collider_extension(entity, *physics_rigid_body);

            if (tn && !tn->Nodes.empty() && !(cs && is_trigger)) {
                // NodesTrigger: compound zone.
                fastgltf::NodeTrigger nt;
                for (const auto ne : tn->Nodes) {
                    if (const auto nit = node_index_of.find(ne); nit != node_index_of.end()) nt.nodes.emplace_back(nit->second);
                }
                physics_rigid_body->trigger = fastgltf::Optional<TriggerVariant>{TriggerVariant{std::move(nt)}};
            }

            if (pj) {
                fastgltf::Joint joint{};
                if (const auto cit = node_index_of.find(pj->ConnectedNode); cit != node_index_of.end()) joint.connectedNode = cit->second;
                if (const auto dit = physics_jointdef_to_index.find(pj->JointDefEntity); dit != physics_jointdef_to_index.end()) joint.joint = dit->second;
                joint.enableCollision = pj->EnableCollision;
                physics_rigid_body->joint = fastgltf::Optional<fastgltf::Joint>{std::move(joint)};
            }
        }

        // Preserve matrix-form sources and emit TRS-form sources from LocalTransform.
        const auto source_matrix = gltf_node ? gltf_node->Matrix : std::nullopt;
        const auto fg_transform = [&]() -> std::variant<fastgltf::TRS, fastgltf::math::fmat4x4> {
            if (source_matrix) return std::bit_cast<fastgltf::math::fmat4x4>(*source_matrix);
            return fastgltf::TRS{
                .translation = std::bit_cast<fastgltf::math::fvec3>(local_transform.P),
                .rotation = std::bit_cast<fastgltf::math::fquat>(local_transform.R),
                .scale = std::bit_cast<fastgltf::math::fvec3>(local_transform.S),
            };
        }();

        // KHR_audio_rigid_bodies: emit the node's modal model, acoustic surface, and instance.
        fastgltf::Optional<fastgltf::AudioRigidBody> audio_rigid_body;
        const auto *modes = r.try_get<const ModalModes>(entity);
        const bool has_modal_model = modes && !modes->Freqs.empty();
        const auto *contact_surface = r.try_get<const ContactSurface>(entity);
        const bool has_acoustic_surface = contact_surface != nullptr;
        // The derivation material lives on the node, and both the model and the surface reference it.
        const auto *acoustic_material = has_modal_model || has_acoustic_surface ? r.try_get<const AcousticMaterial>(entity) : nullptr;
        const auto acoustic_material_index = dedupe(audio_rigid_body_materials, acoustic_material, [&] {
            const auto &p = acoustic_material->Properties;
            asset.acousticMaterials.emplace_back(fastgltf::AcousticMaterial{
                .density = p.Density,
                .youngsModulus = p.YoungModulus,
                .poissonRatio = p.PoissonRatio,
                .alpha = p.Alpha,
                .beta = p.Beta,
                .name = ToFgStr(acoustic_material->Name),
            });
            return asset.acousticMaterials.size() - 1;
        });

        const auto acoustic_surface_index = dedupe(audio_rigid_body_surfaces, contact_surface, [&] {
            fastgltf::AcousticSurface out{
                .roughness = contact_surface->Roughness,
                .correlationLength = contact_surface->CorrelationLength,
                .waviness = contact_surface->Waviness,
                .wavinessLength = contact_surface->WavinessLength,
                .spectralSlope = contact_surface->SpectralSlope,
                .shortWavelength = contact_surface->ShortWavelength,
                .profile = {},
                .sampleSpacing = {},
                .normalTexture = {},
                .material = acoustic_material_index,
                .name = ToFgStr(contact_surface->Name),
            };
            if (contact_surface->HasMeasuredProfile()) {
                out.profile = AddDataAccessor(std::span<const float>(contact_surface->Profile), fastgltf::AccessorType::Scalar, fastgltf::ComponentType::Float);
                out.sampleSpacing = contact_surface->SampleSpacing;
            }
            // Omit texture references removed by scene edits.
            if (const auto &nt = contact_surface->NormalTexture; nt && nt->Texture < asset.textures.size()) {
                out.normalTexture = ToFgNormalTexInfo({.Slot = nt->Texture, .TexCoord = nt->TexCoord}, nt->Scale, nullptr, uses_texture_transform);
            }
            asset.acousticSurfaces.emplace_back(std::move(out));
            return asset.acousticSurfaces.size() - 1;
        });

        if (has_modal_model) {
            const uint32_t n_modes = modes->Freqs.size(), n_points = modes->Positions.size();

            // decayRates d = ln(1000)/T60 (T60 == 0 is the undamped sentinel, d = 0).
            std::vector<float> decay_rates(n_modes);
            for (uint32_t m = 0; m < n_modes; ++m) decay_rates[m] = modes->T60s[m] > 0 ? float(Ln1000 / modes->T60s[m]) : 0.f;

            // shapes are mode-major on the wire: element m*P + i is mode m at sample point i.
            std::vector<vec3> shapes(n_modes * n_points);
            for (uint32_t m = 0; m < n_modes; ++m) {
                for (uint32_t i = 0; i < n_points; ++i) shapes[m * n_points + i] = modes->Shapes[i][m];
            }

            fastgltf::ModalModel model{
                .frequencies = AddDataAccessor(std::span<const float>(modes->Freqs), fastgltf::AccessorType::Scalar, fastgltf::ComponentType::Float),
                .decayRates = AddDataAccessor(std::span<const float>(decay_rates), fastgltf::AccessorType::Scalar, fastgltf::ComponentType::Float),
                .positions = AddDataAccessor(std::span<const vec3>(modes->Positions), fastgltf::AccessorType::Vec3, fastgltf::ComponentType::Float),
                .shapes = AddDataAccessor(std::span<const vec3>(shapes), fastgltf::AccessorType::Vec3, fastgltf::ComponentType::Float),
                .indices = modes->Indices.empty() ? fastgltf::Optional<size_t>{} : fastgltf::Optional<size_t>{AddDataAccessor(std::span<const uint32_t>(modes->Indices), fastgltf::AccessorType::Scalar, fastgltf::ComponentType::UnsignedInt)},
                .material = acoustic_material_index,
                .massProperties = {},
                .name = ToFgStr(node_name),
            };
            if (const auto *mp = r.try_get<const MassProperties>(entity)) {
                // Scales stored mass and inertia from solved density to the mesh's current material density.
                const double rho_ratio = ModalDensityRatio(r, entity);
                const auto &q = mp->InertiaOrientation;
                const vec3 inertia = mp->InertiaDiagonal * float(rho_ratio);
                model.massProperties = fastgltf::ModalMassProperties{
                    .mass = fastgltf::num(mp->Mass * rho_ratio),
                    .centerOfMass = std::bit_cast<fastgltf::math::fvec3>(mp->CenterOfMass),
                    .inertiaDiagonal = std::bit_cast<fastgltf::math::fvec3>(inertia),
                    .inertiaOrientation = std::bit_cast<fastgltf::math::fvec4>(q),
                };
            }

            const auto *gain = r.try_get<const ModalGain>(entity);
            audio_rigid_body = fastgltf::AudioRigidBody{
                .modalModel = asset.modalModels.size(),
                .acousticSurface = acoustic_surface_index,
                .gain = fastgltf::num(gain ? gain->Value : 1.f),
            };
            asset.modalModels.emplace_back(std::move(model));
        } else if (acoustic_surface_index.has_value()) {
            // A body that only supplies its finish to contacts against it, such as a floor.
            audio_rigid_body = fastgltf::AudioRigidBody{.modalModel = {}, .acousticSurface = acoustic_surface_index};
        }

        // A node with a Visibility flag exports that flag at rest. Other nodes derive it from their hidden state.
        bool visible = !export_node.Hidden || (export_node.Parent && nodes[*export_node.Parent].Hidden);
        if (const auto owner = export_node.Instances.empty() ? state::Null : export_node.Instances.front(); owner != state::Null) {
            if (const auto *visibility = r.try_get<const Visibility>(owner)) {
                auto flag = *visibility;
                OverlayRest(r, owner, state::Key<Visibility>(), 0, std::as_writable_bytes(std::span{&flag, 1}));
                visible = flag.Visible != 0;
            }
        }
        asset.nodes.emplace_back(fastgltf::Node{
            .meshIndex = mesh_index,
            .skinIndex = skin_index,
            .cameraIndex = camera_index,
            .lightIndex = light_index,
            .children = std::move(children),
            .weights = {},
            .transform = fg_transform,
            .instancingAttributes = std::move(instancing),
            .name = ToFgStr(node_name),
            .physicsRigidBody = std::move(physics_rigid_body),
            .audioRigidBody = audio_rigid_body,
            .visible = visible,
            .selectable = true,
            .hoverable = true,
        });
    }

    // Preserves one EXT_lights_image_based resource on the default scene.
    // Per-scene IBL assignment remains outside the round-trip model.
    fastgltf::Optional<size_t> default_scene_ibl_index;
    if (sa.ImageBasedLight) {
        const auto &src_ibl = *sa.ImageBasedLight;
        const auto *image_light = r.try_get<const ImageLight>(viewport);
        auto light = image_light ? *image_light : ImageLight{};
        OverlayRest(r, viewport, state::Key<ImageLight>(), 0, std::as_writable_bytes(std::span{&light, 1}));
        fastgltf::ImageBasedLight ibl{
            .intensity = light.Intensity,
            .rotation = fastgltf::math::fquat(light.Rotation.x, light.Rotation.y, light.Rotation.z, light.Rotation.w),
            .specularImageSize = src_ibl.SpecularImageSize,
            .specularImages = {},
            .irradianceCoefficients = {},
            .name = ToFgStr(src_ibl.Name),
        };
        ibl.specularImages.reserve(src_ibl.SpecularImageIndicesByMip.size());
        for (const auto &mip : src_ibl.SpecularImageIndicesByMip) {
            std::array<size_t, 6> faces{};
            for (size_t face = 0; face < 6; ++face) faces[face] = mip[face];
            ibl.specularImages.emplace_back(faces);
        }
        if (src_ibl.IrradianceCoefficients) {
            std::array<fastgltf::math::fvec3, 9> coeffs{};
            for (size_t i = 0; i < 9; ++i) {
                const auto &c = (*src_ibl.IrradianceCoefficients)[i];
                coeffs[i] = std::bit_cast<fastgltf::math::fvec3>(c);
            }
            ibl.irradianceCoefficients = coeffs;
        }
        asset.imageBasedLights.emplace_back(std::move(ibl));
        default_scene_ibl_index = size_t{0};
    }

    if (!scenes_ordered.empty()) {
        // Emit every scene, including empty ones (valid glTF). Default = the active scene.
        std::optional<size_t> active_emitted;
        for (const auto se : scenes_ordered) {
            fastgltf::pmr::MaybeSmallVector<size_t> scene_roots;
            for (const auto ni : compute_roots(se)) scene_roots.emplace_back(ni);
            if (se == active_scene) active_emitted = asset.scenes.size();
            asset.scenes.emplace_back(fastgltf::Scene{.nodeIndices = std::move(scene_roots), .imageBasedLightIndex = {}, .name = ToFgStr(r.get<const Scene>(se).Name)});
        }
        asset.defaultScene = active_emitted.value_or(0);
        if (default_scene_ibl_index) asset.scenes[*asset.defaultScene].imageBasedLightIndex = default_scene_ibl_index;
    } else {
        // No scene entities (non-glTF / runtime-built): synthesize a single scene from current roots.
        fastgltf::pmr::MaybeSmallVector<size_t> scene_roots;
        for (const auto ni : compute_roots(state::Null)) scene_roots.emplace_back(ni);
        asset.scenes.emplace_back(fastgltf::Scene{
            .nodeIndices = std::move(scene_roots),
            .imageBasedLightIndex = default_scene_ibl_index,
            .name = {},
        });
        asset.defaultScene = 0;
    }

    asset.extensionsRequired.reserve(sa.ExtensionsRequired.size());
    for (const auto &e : sa.ExtensionsRequired) asset.extensionsRequired.emplace_back(e);

    if (std::ranges::any_of(asset.nodes, [](const auto &n) { return !n.visible; })) asset.extensionsUsed.emplace_back("KHR_node_visibility");
    if (!asset.lights.empty()) asset.extensionsUsed.emplace_back("KHR_lights_punctual");
    if (uses_animation_pointer) asset.extensionsUsed.emplace_back("KHR_animation_pointer");
    if (uses_gpu_instancing) asset.extensionsUsed.emplace_back("EXT_mesh_gpu_instancing");
    if (uses_physics_rigid_bodies || !asset.physicsMaterials.empty() || !asset.collisionFilters.empty() || !asset.physicsJoints.empty()) {
        asset.extensionsUsed.emplace_back("KHR_physics_rigid_bodies");
    }
    if (!asset.shapes.empty()) asset.extensionsUsed.emplace_back("KHR_implicit_shapes");
    if (!asset.modalModels.empty() || !asset.acousticSurfaces.empty()) asset.extensionsUsed.emplace_back("KHR_audio_rigid_bodies");
    if (!asset.imageBasedLights.empty()) asset.extensionsUsed.emplace_back("EXT_lights_image_based");
    const auto any_material = [&](auto pred) { return std::ranges::any_of(asset.materials, pred); };
    if (any_material([](const auto &m) { return m.unlit; })) asset.extensionsUsed.emplace_back("KHR_materials_unlit");
    if (any_material([](const auto &m) { return m.ior.has_value(); })) asset.extensionsUsed.emplace_back("KHR_materials_ior");
    if (any_material([](const auto &m) { return m.emissiveStrength.has_value(); })) asset.extensionsUsed.emplace_back("KHR_materials_emissive_strength");
    if (any_material([](const auto &m) { return m.dispersion.has_value(); })) asset.extensionsUsed.emplace_back("KHR_materials_dispersion");
    if (any_material([](const auto &m) { return m.sheen != nullptr; })) asset.extensionsUsed.emplace_back("KHR_materials_sheen");
    if (any_material([](const auto &m) { return m.specular != nullptr; })) asset.extensionsUsed.emplace_back("KHR_materials_specular");
    if (any_material([](const auto &m) { return m.transmission != nullptr; })) asset.extensionsUsed.emplace_back("KHR_materials_transmission");
    if (any_material([](const auto &m) { return m.diffuseTransmission != nullptr; })) asset.extensionsUsed.emplace_back("KHR_materials_diffuse_transmission");
    if (any_material([](const auto &m) { return m.volume != nullptr; })) asset.extensionsUsed.emplace_back("KHR_materials_volume");
    if (any_material([](const auto &m) { return m.clearcoat != nullptr; })) asset.extensionsUsed.emplace_back("KHR_materials_clearcoat");
    if (any_material([](const auto &m) { return m.anisotropy != nullptr; })) asset.extensionsUsed.emplace_back("KHR_materials_anisotropy");
    if (any_material([](const auto &m) { return m.iridescence != nullptr; })) asset.extensionsUsed.emplace_back("KHR_materials_iridescence");
    if (const auto *mv = r.try_get<const ::MaterialVariants>(viewport); mv && !mv->Names.empty()) {
        asset.materialVariants.reserve(mv->Names.size());
        for (const auto &v : mv->Names) asset.materialVariants.emplace_back(v);
        asset.extensionsUsed.emplace_back("KHR_materials_variants");
    }
    if (uses_texture_transform) asset.extensionsUsed.emplace_back("KHR_texture_transform");
    if (std::ranges::any_of(asset.textures, [](const auto &t) { return t.webpImageIndex.has_value(); })) {
        asset.extensionsUsed.emplace_back("EXT_texture_webp");
    }

    // Finalize buffer. sources::Vector owns our binary blob; FileExporter writes it as a sibling .bin.
    asset.buffers.emplace_back(fastgltf::Buffer{
        .byteLength = bin.size(),
        .data = fastgltf::sources::Vector{.bytes = std::move(bin), .mimeType = fastgltf::MimeType::None},
        .name = ToFgStr(path.stem().string()),
    });

    asset.accessors = std::move(accessors);
    asset.bufferViews = std::move(bufferViews);

    fastgltf::FileExporter exporter;
    if (!sa.ExtrasByEntity.empty()) {
        exporter.setUserPointer(const_cast<ExtrasMap *>(&sa.ExtrasByEntity));
        exporter.setExtrasWriteCallback(EmitExtras);
    }
    const auto ext = path.extension();
    const auto err = ext == ".glb" ? exporter.writeGltfBinary(asset, path) : exporter.writeGltfJson(asset, path);
    if (err != fastgltf::Error::None) {
        return std::unexpected{std::format("fastgltf export to '{}' failed: {}", path.string(), fastgltf::getErrorMessage(err))};
    }
    return {};
}

} // namespace gltf
