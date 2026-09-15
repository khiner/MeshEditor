#include "action/Io.h"
#include "editor/AudioIntegration.h"
#include "state/Scene.h"

#include "CameraTypes.h"
#include "Profile.h"
#include "Variant.h"
#include "action/Errors.h"
#include "animation/AnimationTimeline.h"
#include "audio/AcousticMaterial.h"
#include "audio/AudioSystem.h"
#include "audio/ModalModelFile.h"
#include "audio/RealImpact.h"
#include "audio/RealImpactComponents.h"
#include "gltf/GltfScene.h"
#include "mesh/MeshBatch.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "numeric/Angles.h"
#include "object/ObjectOps.h"
#include "render/GpuBufferOps.h"
#include "scene/Defaults.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/Viewport.h"

#include <format>
#include <numbers>
#include <utility>

using std::ranges::to;

namespace action::io {
namespace {
// Load a glTF/glb and apply its camera/animation side effects.
void LoadGltfFile(state::Scene &r, state::Entity viewport, const std::filesystem::path &path) {
    const profile::CpuScope scope{"LoadGltfFile"};
    auto &c = r.ctx();
    auto result = gltf::LoadGltf(path, {r, viewport, c.get<mtl::BindlessSet>(), c.get<GpuBuffers>(), c.get<MeshStore>(), c.get<TextureStore>(), c.get<EnvironmentStore>()});
    if (!result) {
        Fail(r, std::format("Error loading glTF file '{}': {}", path.string(), result.error()));
        return;
    }

    if (result->FirstCameraObject != state::Null) SetLookThrough(r, viewport, result->FirstCameraObject);
    if (result->ImportedAnimation) {
        JumpToStartFrame(r, viewport);
        r.edit<LastEvaluatedFrame>(viewport).Value = -1;
    }
}
} // namespace

void Apply(state::Scene &r, state::Entity viewport, const Action &action) {
    std::visit(
        overloaded{
            [&](const LoadDefaultScene &) { AddDefaultSceneContent(r); },
            [&](const Load &a) {
                const auto &path = a.Path;
                const auto ext = path.extension().string();
                if (ext == ".gltf" || ext == ".glb") LoadGltfFile(r, viewport, a.Path);
                else if (ext == ".obj" || ext == ".ply") RequestImportMesh(r, viewport, path, MeshInstanceCreateInfo{.Name = path.stem().string()});
                else Fail(r, std::format("Unsupported file format: '{}'", ext));
            },
            [&](const SaveGltf &a) {
                auto &c = r.ctx();
                if (auto save = gltf::SaveGltf(a.Path, {r, viewport, c.get<GpuBuffers>(), c.get<MeshStore>(), c.get<TextureStore>(), &c.get<const mtl::Context>(), &GetBufferContext(r)}); !save) {
                    Fail(r, std::format("Error saving glTF file '{}': {}", a.Path.string(), save.error()));
                }
            },
            [&](const LoadGltf &a) { LoadGltfFile(r, viewport, a.Path); },
            [&](const LoadRealImpact &a) {
                auto source = RealImpact::LoadSource(r, a.Path);
                if (!source) {
                    Fail(r, std::move(source.error()));
                    return;
                }

                ClearMeshes(r, viewport);
                const auto [mesh_entity, instance_entity] = ImportMesh(
                    r, viewport,
                    source->Mesh,
                    MeshInstanceCreateInfo{.Name = std::move(source->Name), .Transform = {.R = RealImpact::ObjectRotationToYUp}},
                    true // Weld vertices
                );

                // The npy file's vertex indices use the source OBJ numbering, so look up by position instead.
                std::vector<uint32_t> vertex_indices(RealImpact::NumImpactVertices);
                {
                    const auto &impact_positions = source->Positions;
                    const auto &mesh = GetMesh(r, mesh_entity);
                    for (size_t i = 0; i < impact_positions.size(); ++i) {
                        vertex_indices[i] = *mesh.FindNearestVertex(impact_positions[i]);
                    }
                }

                const auto &listener_points = source->Listeners;
                const auto created = CreateMesh(r, {.Data = primitive::CreateMesh({primitive::Cylinder{0.5f * RealImpact::MicWidthMm / 1000.f, RealImpact::MicLengthMm / 1000.f}}), .FlatShaded = true});
                const auto [listener_mesh_entity, _] = ::AddMesh(r, created.StoreId);
                for (const auto &listener_point : listener_points) {
                    static const auto rot_z = numeric::AngleAxis(std::numbers::pi_v<float> / 2.f, vec3{0, 0, 1}); // Cylinder's center is along the Y axis.
                    const auto listener_instance_entity = ::AddMeshInstance(
                        r, listener_mesh_entity,
                        {
                            .Name = std::format("RealImpact Microphone: {}", listener_point.Index),
                            .Transform = {
                                .P = listener_point.GetPosition(Defaults::WorldUp, true),
                                .R = numeric::AngleAxis(numeric::Radians(float(listener_point.AngleDeg)), Defaults::WorldUp) * rot_z,
                            },
                            .Select = MeshInstanceCreateInfo::SelectBehavior::None,
                        }
                    );
                    r.emplace<RealImpactMicrophone>(listener_instance_entity, listener_point.Index);

                    if (listener_point.Index == RealImpact::CenteredListenerIndex) {
                        r.emplace<RealImpactActiveMicrophone>(instance_entity, listener_instance_entity);

                        if (const auto material_name = RealImpact::FindMaterialName(r.get<Name>(instance_entity).Value)) {
                            if (const auto *material = materials::acoustic::Find(*material_name)) r.emplace<AcousticMaterial>(instance_entity, *material);
                        }
                        r.emplace<ScaleLocked>(instance_entity);
                        r.emplace<RealImpactVertices>(instance_entity, vertex_indices, source->Samples);
                        if (source->Samples.empty()) continue;
                        auto samples = RealImpact::LoadSamples(r, source->Samples, listener_point.Index);
                        if (!samples) {
                            Fail(r, std::move(samples.error()));
                            return;
                        }
                        SetVertexSamples(r, instance_entity, vertex_indices, *samples);
                    }
                }
            },
        },
        action
    );
}
} // namespace action::io
