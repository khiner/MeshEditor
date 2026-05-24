#include "action/Io.h"
#include "AnimationTimeline.h"
#include "Defaults.h"
#include "GpuBuffers.h"
#include "ObjectOps.h"
#include "SceneGraph.h"
#include "Timer.h"
#include "ViewportOps.h"
#include "audio/AudioSystem.h"
#include "audio/RealImpact.h"
#include "gltf/GltfScene.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "physics/PhysicsWorld.h"

using std::ranges::find_if;
using std::ranges::to;

namespace {
void NewDefaultScene(entt::registry &r, entt::entity viewport) {
    ClearMeshes(r, viewport);

    auto &meshes = r.ctx().get<MeshStore>();
    auto &buffers = r.ctx().get<GpuBuffers>();

    constexpr PrimitiveShape default_shape{primitive::Cuboid{}};
    const auto [mesh_entity, _] = ::AddMesh(r, meshes, meshes.CreateMesh(primitive::CreateMesh(default_shape), {}, {}), MeshInstanceCreateInfo{.Name = ToString(default_shape)});
    r.emplace<PrimitiveShape>(mesh_entity, default_shape);

    // startup.blend data, in Blender's frame (Z-up, -Y forward)
    constexpr vec3 LightLoc{4.07625, 1.00545, 5.90386}, CameraLoc{7.358891, -6.925791, 4.958309}, CameraEulerXYZ{1.109319, 0, 0.815801};
    constexpr float Lens{50}, SensorX{36}, RenderW{16}, RenderH{9};
    // Blender Z-up -> MeshEditor Y-up is a -90° rotation about +X: (x, y, z) -> (x, z, -y)
    const auto to_y_up_pos = [](vec3 v) { return vec3{v.x, v.z, -v.y}; };
    const quat to_y_up_rot = glm::angleAxis(-float(M_PI_2), vec3{1, 0, 0});
    // Matches Blender glTF exporter (cameras.py / yvof_blender_to_gltf): horizontal fit since render aspect > sensor aspect
    const float hfov = 2 * std::atan(SensorX / (2 * Lens));
    const float yfov = 2 * std::atan(std::tan(hfov * 0.5) * RenderH / RenderW);

    ::AddLight(r, meshes, buffers, ObjectCreateInfo{.Name = "Light", .Transform = {.P = to_y_up_pos(LightLoc)}, .Select = MeshInstanceCreateInfo::SelectBehavior::None});
    ::AddCamera(r, meshes, buffers, ObjectCreateInfo{.Name = "Camera", .Transform = {.P = to_y_up_pos(CameraLoc), .R = to_y_up_rot * quat{CameraEulerXYZ}}, .Select = MeshInstanceCreateInfo::SelectBehavior::None}, Perspective{.FieldOfViewRad = yfov, .FarClip = 1000, .NearClip = DefaultPerspectiveNearClip});
}
} // namespace

namespace action::io {
void Apply(entt::registry &r, entt::entity viewport, const Action &action) {
    std::visit(overloaded{[&](NewDefaultScene) { ::NewDefaultScene(r, viewport); }}, action);
}

std::expected<void, std::string> Apply(entt::registry &r, entt::entity viewport, const FallibleActions &action) {
    const auto &vk = r.ctx().get<const VulkanResources>();
    auto &slots = r.ctx().get<DescriptorSlots>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &meshes = r.ctx().get<MeshStore>();
    auto &textures = r.ctx().get<TextureStore>();
    auto &environments = r.ctx().get<EnvironmentStore>();
    auto &physics = r.ctx().get<PhysicsWorld>();
    return std::visit(
        overloaded{
            [&](const SaveGltf &a) -> std::expected<void, std::string> {
                return gltf::SaveGltf(a.Path, {r, viewport, buffers, meshes, textures, &vk, &buffers.Ctx});
            },
            [&](const LoadGltf &a) -> std::expected<void, std::string> {
                const Timer timer{"LoadGltf"};
                auto result = gltf::LoadGltf(a.Path, {r, viewport, slots, buffers, meshes, textures, environments});
                if (!result) return std::unexpected{std::move(result.error())};

                // TODO: drive reactively from track<changes::PhysicsShape>.
                if (!r.view<ColliderShape>().empty()) physics.RecomputeSceneScale(r);

                if (result->FirstCameraObject != entt::null) {
                    const auto camera_entity = result->FirstCameraObject;
                    SetLookThrough(r, viewport, camera_entity);
                    const auto &wt = r.get<WorldTransform>(camera_entity);
                    r.replace<ViewCamera>(viewport, ViewCamera{wt.P, wt.P + CameraForward(wt), r.get<Camera>(camera_entity)});
                }
                if (result->ImportedAnimation) {
                    JumpToStartFrame(r, viewport);
                    r.get<LastEvaluatedFrame>(viewport).Value = -1;
                }
                r.emplace_or_replace<ProfileNextProcessComponentEvents>(viewport);
                return {};
            },
            [&](const LoadRealImpact &a) -> std::expected<void, std::string> {
                auto object_name = RealImpact::ValidateDirectory(a.Directory);
                if (!object_name) return std::unexpected(std::move(object_name.error()));

                ClearMeshes(r, viewport);
                const auto [mesh_entity, instance_entity] = ImportMesh(
                    r,
                    a.Directory / "transformed.obj",
                    MeshInstanceCreateInfo{
                        .Name = std::move(*object_name),
                        .Transform = {.R = RealImpact::ObjectRotationToYUp},
                    }
                );

                // Ignore the npy file's vertex indices: deduplication may have invalidated them. Look up by position instead.
                std::vector<uint32_t> vertex_indices(RealImpact::NumImpactVertices);
                {
                    const auto impact_positions = RealImpact::LoadPositions(a.Directory);
                    const auto &mesh = r.get<Mesh>(mesh_entity);
                    for (size_t i = 0; i < impact_positions.size(); ++i) {
                        vertex_indices[i] = *mesh.FindNearestVertex(impact_positions[i]);
                    }
                }

                const auto listener_points = RealImpact::LoadListenerPoints(a.Directory);
                const auto [listener_mesh_entity, _] = ::AddMesh(r, meshes, meshes.CreateMesh(primitive::CreateMesh({primitive::Cylinder{0.5f * RealImpact::MicWidthMm / 1000.f, RealImpact::MicLengthMm / 1000.f}}), {}, {}));
                for (const auto &listener_point : listener_points) {
                    static const auto rot_z = glm::angleAxis(float(M_PI_2), vec3{0, 0, 1}); // Cylinder's center is along the Y axis.
                    const auto listener_instance_entity = ::AddMeshInstance(
                        r, listener_mesh_entity,
                        {
                            .Name = std::format("RealImpact Microphone: {}", listener_point.Index),
                            .Transform = {
                                .P = listener_point.GetPosition(Defaults::World.Up, true),
                                .R = glm::angleAxis(glm::radians(float(listener_point.AngleDeg)), Defaults::World.Up) * rot_z,
                            },
                            .Select = MeshInstanceCreateInfo::SelectBehavior::None,
                        }
                    );
                    r.emplace<RealImpactMicrophone>(listener_instance_entity, listener_point.Index);

                    if (listener_point.Index == RealImpact::CenteredListenerIndex) {
                        r.emplace<RealImpactActiveMicrophone>(instance_entity, listener_instance_entity);

                        auto material_name = RealImpact::FindMaterialName(r.get<Name>(instance_entity).Value);
                        if (const auto real_impact_material = material_name ?
                                find_if(materials::acoustic::All, [name = *material_name](const AcousticMaterial &m) { return m.Name == name; }) :
                                std::ranges::end(materials::acoustic::All)) {
                            r.emplace<AcousticMaterial>(mesh_entity, *real_impact_material);
                        }
                        r.emplace<ScaleLocked>(instance_entity);
                        r.emplace<RealImpactVertices>(instance_entity, vertex_indices);
                        SetVertexSamples(r, viewport, instance_entity, vertex_indices, to<std::vector>(RealImpact::LoadSamples(a.Directory, listener_point.Index)));
                    }
                }
                return {};
            },
        },
        action
    );
}
} // namespace action::io
