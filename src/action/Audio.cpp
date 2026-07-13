#include "action/Audio.h"
#include "Path.h"
#include "audio/AudioSystem.h"
#include "audio/RealImpact.h"
#include "audio/SoundVertices.h"
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
            [&](SetupModalModel) {
                const auto e = FindActiveEntity(r);
                if (!r.all_of<ModalSolveSettings>(e)) r.emplace<ModalSolveSettings>(e);
                if (const auto mesh_e = GetActiveMeshEntity(r); !r.all_of<AcousticMaterial>(mesh_e)) r.emplace<AcousticMaterial>(mesh_e, materials::acoustic::All.front());
            },
            [&](const ApplyModalModel &a) { ::ApplyModalModel(r, a.SoundEntity, a.Path); },
            [&](const AssignVertexSamples &a) {
                auto frames = LoadAudioFrames(a.Path.string(), DeviceSampleRate(r));
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
            [&]<typename T>(const Replace<T> &a) { r.emplace_or_replace<T>(a.Entity, a.Value); },
        },
        action
    );
}
} // namespace action::audio
