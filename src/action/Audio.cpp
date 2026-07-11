#include "action/Audio.h"
#include "Path.h"
#include "audio/AudioSystem.h"
#include "audio/RealImpact.h"
#include "audio/SoundVertices.h"
#include "audio/mesh2modes.h"
#include "render/Instance.h"
#include "scene/Entity.h"
#include <entt/entity/registry.hpp>

using std::ranges::to;

namespace action::audio {
void Apply(entt::registry &r, entt::entity viewport, const Action &action) {
    std::visit(
        overloaded{
            [&](const ApplyExciteImpact &a) {
                r.emplace_or_replace<MeshActiveElement>(r.get<Instance>(a.InstanceEntity).Entity, a.VertexIndex);
                r.emplace_or_replace<VertexForce>(a.InstanceEntity, a.VertexIndex, 1.f);
            },
            [&](ClearExciteImpacts) { r.clear<VertexForce>(); },
            [&](const SetModel &a) { ::SetModel(r, FindActiveEntity(r), a.Model); },
            [&](const SetExciteVertex &a) {
                const auto e = FindActiveEntity(r);
                r.remove<VertexForce>(e);
                r.emplace_or_replace<MeshActiveElement>(GetActiveMeshEntity(r), a.MeshVertex);
                ::Stop(r, e);
            },
            [&](const StartExcite &a) {
                const auto e = FindActiveEntity(r);
                r.remove<VertexForce>(e);
                r.emplace<VertexForce>(e, a.Vertex, 1.f);
            },
            [&](StopExcite) { r.remove<VertexForce>(FindActiveEntity(r)); },
            [&](DeleteSoundObject) { RemoveAudioComponents(r, FindActiveEntity(r)); },
            [&](const StartRecording &a) { r.emplace_or_replace<Recording>(FindActiveEntity(r), a.FrameCount); },
            [&](const OpenModalForm &a) { r.emplace_or_replace<ModalModelCreateInfo>(FindActiveEntity(r), *a.Info); },
            [&](CancelModalForm) { r.remove<ModalModelCreateInfo>(FindActiveEntity(r)); },
            [&](SubmitModalForm) {
                const auto e = FindActiveEntity(r);
                ::Stop(r, e);
                r.emplace_or_replace<AcousticMaterial>(GetActiveMeshEntity(r), r.get<const ModalModelCreateInfo>(e).Material);
                r.remove<ModalModelCreateInfo>(e);
            },
            [&](const ApplyModalModel &a) { ::ApplyModalModel(r, a.SoundEntity, a.Path); },
            [&](const RescaleModalModel &a) {
                const auto e = FindActiveEntity(r);
                ::Stop(r, e);
                const auto mesh_entity = GetActiveMeshEntity(r);
                const auto *summary = r.try_get<const ModalEigenSummary>(e);
                const auto *modes = r.try_get<const ModalModes>(e);
                if (summary && modes) {
                    if (auto rescaled = modal::RescaleModes(*summary, *modes, a.Material->Properties, {.FundamentalFreq = a.FundamentalFreq})) {
                        // Mass and inertia are linear in density, so they rescale from the mesh's current material.
                        if (const auto *mat = r.try_get<const AcousticMaterial>(mesh_entity)) {
                            if (const auto *mp = r.try_get<const MassProperties>(e)) {
                                const double rho_ratio = a.Material->Properties.Density / mat->Properties.Density;
                                auto scaled_mp = *mp;
                                scaled_mp.Mass *= rho_ratio;
                                scaled_mp.InertiaDiagonal *= float(rho_ratio);
                                r.emplace_or_replace<MassProperties>(e, scaled_mp);
                            }
                        }
                        r.emplace_or_replace<ModalModes>(e, std::move(*rescaled));
                    }
                }
                r.emplace_or_replace<AcousticMaterial>(mesh_entity, *a.Material);
                r.remove<ModalModelCreateInfo>(e);
            },
            [&](const AssignVertexSamples &a) {
                auto frames = LoadAudioFrames(a.Path.string());
                if (!frames.empty()) ::AssignVertexSample(r, viewport, FindActiveEntity(r), a.MeshVertices, a.Path, std::move(frames));
            },
            [&](const ActivateRealImpactMicrophone &a) {
                const auto dir = r.get<const Path>(r.get<const Instance>(a.TargetSoundEntity).Entity).Value.parent_path();
                const auto &vertex_indices = r.get<const RealImpactVertices>(a.TargetSoundEntity).Vertices;
                const auto mic_index = r.get<const RealImpactMicrophone>(a.MicrophoneEntity).Index;
                ::SetVertexSamples(r, viewport, a.TargetSoundEntity, vertex_indices, RealImpact::LoadSamples(dir, mic_index) | to<std::vector>());
                r.emplace_or_replace<RealImpactActiveMicrophone>(a.TargetSoundEntity, a.MicrophoneEntity);
            },
            [&](const RemoveVertexSamples &a) { ::RemoveVertexSamples(r, viewport, FindActiveEntity(r), a.MeshVertices); },
            [&](const SetModalFormMaterial &a) { r.patch<ModalModelCreateInfo>(FindActiveEntity(r), [&](auto &info) { info.Material = *a.Material; }); },
            [&]<typename T>(const Replace<T> &a) { r.emplace_or_replace<T>(a.Entity, a.Value); },
        },
        action
    );
}
} // namespace action::audio
