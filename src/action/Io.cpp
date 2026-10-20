#include "action/Io.h"
#include "editor/AudioIntegration.h"
#include "state/Registry.h"
#include "viewport/ViewportScene.h"

#include "Profile.h"
#include "Variant.h"
#include "action/Errors.h"
#include "audio/AcousticMaterial.h"
#include "audio/RealImpact.h"
#include "audio/RealImpactComponents.h"
#include "gltf/GltfScene.h"
#include "mesh/MeshCreate.h"
#include "mesh/Primitives.h"
#include "numeric/Angles.h"
#include "object/ObjectOps.h"
#include "scene/Defaults.h"
#include "timeline/Timeline.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportEvents.h"

#include <format>

using numeric::Radians;

namespace action::io {
namespace {
void LoadGltfFile(state::Registry &r, const std::filesystem::path &path) {
    const profile::CpuScope scope{"LoadGltfFile"};
    auto result = gltf::LoadGltf(path, r);
    if (!result) {
        Fail(r, std::format("Error loading glTF file '{}': {}", path.string(), result.error().Message), result.error().UnsupportedExtensions);
        return;
    }

    // A live load looks through the first camera, which the active viewport takes in its next frame step.
    if (result->FirstCameraObject != state::Null) r.get_or_emplace<PendingViewSetup>(state::Document).LookThroughEntity = result->FirstCameraObject;
    if (result->ImportedAnimation) {
        JumpToStartFrame(r);
        r.edit<LastEvaluatedFrame>(state::Document).Value = -1;
    }
}
} // namespace

void Apply(state::Registry &r, const Action &action) {
    // A document starts with one scene shown by one viewport.
    const auto add_scene = [&] {
        const auto scene = r.create();
        r.emplace<Scene>(scene);
        EmplaceUniqueName(r, scene, "Scene");
        const auto viewport = r.create();
        r.emplace<ViewportScene>(viewport, scene);
        EmplaceUniqueName(r, viewport, "Viewport");
    };
    std::visit(
        overloaded{
            [&](const LoadDefaultScene &) { add_scene(); AddDefaultSceneContent(r); },
            [&](const LoadEmptyScene &) { add_scene(); },
            [&](const Load &a) {
                const auto &path = a.Path;
                const auto ext = path.extension().string();
                if (ext == ".gltf" || ext == ".glb") LoadGltfFile(r, a.Path);
                else if (ext == ".obj" || ext == ".ply") RequestImportMesh(r, path, MeshInstanceCreateInfo{.Name = path.stem().string()});
                else Fail(r, std::format("Unsupported file format: '{}'", ext));
            },
            [&](const SaveGltf &a) {
                if (auto save = gltf::SaveGltf(a.Path, r); !save) {
                    Fail(r, std::format("Error saving glTF file '{}': {}", a.Path.string(), save.error()));
                }
            },
            [&](const LoadGltf &a) { LoadGltfFile(r, a.Path); },
            [&](const LoadRealImpact &a) {
                auto source = RealImpact::LoadSource(r, a.Path);
                if (!source) {
                    Fail(r, std::move(source.error()));
                    return;
                }

                ClearMeshes(r);
                const auto imported = ImportMesh(
                    r,
                    source->Mesh,
                    MeshInstanceCreateInfo{.Name = std::move(source->Name), .Transform = {.R = RealImpact::ObjectRotationToYUp}},
                    true // Weld vertices
                );

                if (!imported) {
                    Fail(r, imported.error());
                    return;
                }
                const auto [mesh_entity, instance_entity] = *imported;

                // The npy file's vertex indices use the source OBJ numbering, so look up by position instead.
                std::vector<uint32_t> vertex_indices(RealImpact::NumImpactVertices);
                {
                    const auto &impact_positions = source->Positions;
                    const auto &mesh = GetMesh(r, mesh_entity);
                    for (size_t i = 0; i < impact_positions.size(); ++i) {
                        vertex_indices[i] = mesh.VertexOrdinal(mesh.FindNearestVertex(impact_positions[i]));
                    }
                }

                const auto &listener_points = source->Listeners;
                const auto created = CreateMesh(r, {.Data = primitive::CreateMesh({primitive::Cylinder{0.5f * RealImpact::MicWidthMm / 1000.f, RealImpact::MicLengthMm / 1000.f}}), .FlatShaded = true});
                const auto listener_mesh_entity = AddMeshData(r, created.StoreId);
                for (const auto &listener_point : listener_points) {
                    static const auto rot_z = AngleAxis(std::numbers::pi_v<float> / 2.f, vec3{0, 0, 1}); // Cylinder's center is along the Y axis.
                    const auto listener_instance_entity = ::AddMeshInstance(
                        r, listener_mesh_entity,
                        {
                            .Name = std::format("RealImpact Microphone: {}", listener_point.Index),
                            .Transform = {
                                .P = listener_point.GetPosition(Defaults::WorldUp, true),
                                .R = AngleAxis(Radians(float(listener_point.AngleDeg)), Defaults::WorldUp) * rot_z,
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
