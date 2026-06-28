#include "action/Io.h"
#include "CameraTypes.h"
#include "Timer.h"
#include "Variant.h"
#include "action/Errors.h"
#include "animation/AnimationTimeline.h"
#include "audio/AudioSystem.h"
#include "audio/RealImpact.h"
#include "audio/RealImpactComponents.h"
#include "gltf/GltfScene.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "object/ObjectOps.h"
#include "render/GpuBufferOps.h"
#include "scene/Defaults.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportOps.h"

#include <entt/entity/registry.hpp>

using std::ranges::find_if, std::ranges::to;

namespace action::io {
void Apply(entt::registry &r, entt::entity viewport, const Action &action) {
    const auto fail = [&](std::string message) { r.ctx().get<Errors>().Messages.push_back(std::move(message)); };
    std::visit(
        overloaded{
            [&](const Clear &) { ClearScene(r, viewport); },
            [&](const SaveGltf &a) {
                auto &c = r.ctx();
                if (auto save = gltf::SaveGltf(a.Path, {r, viewport, c.get<GpuBuffers>(), c.get<MeshStore>(), c.get<TextureStore>(), &c.get<const VulkanResources>(), &GetBufferContext(r)}); !save) {
                    fail(std::format("Error saving glTF file '{}': {}", a.Path, save.error()));
                }
            },
            [&](const LoadGltf &a) {
                const Timer timer{"LoadGltf"};
                auto &c = r.ctx();
                auto result = gltf::LoadGltf(a.Path, {r, viewport, c.get<DescriptorSlots>(), c.get<GpuBuffers>(), c.get<MeshStore>(), c.get<TextureStore>(), c.get<EnvironmentStore>()});
                if (!result) {
                    fail(std::format("Error loading glTF file '{}': {}", a.Path, result.error()));
                    return;
                }

                if (result->FirstCameraObject != entt::null) SetLookThrough(r, viewport, result->FirstCameraObject);
                if (result->ImportedAnimation) {
                    JumpToStartFrame(r, viewport);
                    r.get<LastEvaluatedFrame>(viewport).Value = -1;
                }
                r.emplace_or_replace<ProfileNextProcessComponentEvents>(viewport);
            },
            [&](const LoadRealImpact &a) {
                const std::filesystem::path directory{a.Directory};
                auto object_name = RealImpact::ValidateDirectory(directory);
                if (!object_name) {
                    fail(std::move(object_name.error()));
                    return;
                }

                auto &meshes = r.ctx().get<MeshStore>();
                ClearMeshes(r, viewport);
                const auto [mesh_entity, instance_entity] = ImportMesh(
                    r,
                    directory / "transformed.obj",
                    MeshInstanceCreateInfo{.Name = std::move(*object_name), .Transform = {.R = RealImpact::ObjectRotationToYUp}},
                    true // Deduplicate vertices by position
                );

                // Ignore the npy file's vertex indices: deduplication may have invalidated them. Look up by position instead.
                std::vector<uint32_t> vertex_indices(RealImpact::NumImpactVertices);
                {
                    const auto impact_positions = RealImpact::LoadPositions(directory);
                    const auto &mesh = GetMesh(r, mesh_entity);
                    for (size_t i = 0; i < impact_positions.size(); ++i) {
                        vertex_indices[i] = *mesh.FindNearestVertex(impact_positions[i]);
                    }
                }

                const auto listener_points = RealImpact::LoadListenerPoints(directory);
                const auto [listener_mesh_entity, _] = ::AddMesh(r, meshes, meshes.CreateMesh(primitive::CreateMesh({primitive::Cylinder{0.5f * RealImpact::MicWidthMm / 1000.f, RealImpact::MicLengthMm / 1000.f}}), {}, {}));
                for (const auto &listener_point : listener_points) {
                    static const auto rot_z = glm::angleAxis(float(M_PI_2), vec3{0, 0, 1}); // Cylinder's center is along the Y axis.
                    const auto listener_instance_entity = ::AddMeshInstance(
                        r, listener_mesh_entity,
                        {
                            .Name = std::format("RealImpact Microphone: {}", listener_point.Index),
                            .Transform = {
                                .P = listener_point.GetPosition(Defaults::WorldUp, true),
                                .R = glm::angleAxis(glm::radians(float(listener_point.AngleDeg)), Defaults::WorldUp) * rot_z,
                            },
                            .Select = MeshInstanceCreateInfo::SelectBehavior::None,
                        }
                    );
                    r.emplace<RealImpactMicrophone>(listener_instance_entity, listener_point.Index);

                    if (listener_point.Index == RealImpact::CenteredListenerIndex) {
                        r.emplace<RealImpactActiveMicrophone>(instance_entity, listener_instance_entity);

                        auto material_name = RealImpact::FindMaterialName(r.get<Name>(instance_entity).Value);
                        if (const auto *const real_impact_material = material_name ?
                                find_if(materials::acoustic::All, [name = *material_name](const AcousticMaterial &m) { return m.Name == name; }) :
                                std::ranges::end(materials::acoustic::All)) {
                            r.emplace<AcousticMaterial>(mesh_entity, *real_impact_material);
                        }
                        r.emplace<ScaleLocked>(instance_entity);
                        r.emplace<RealImpactVertices>(instance_entity, vertex_indices);
                        SetVertexSamples(r, viewport, instance_entity, vertex_indices, to<std::vector>(RealImpact::LoadSamples(directory, listener_point.Index)));
                    }
                }
            },
        },
        action
    );
}
} // namespace action::io
