#include "action/Audio.h"
#include "Path.h"
#include "action/Errors.h"
#include "audio/AudioSystem.h"
#include "audio/RealImpact.h"
#include "audio/SoundVertices.h"
#include "editor/AudioIntegration.h"
#include "render/Instance.h"
#include "scene/Entity.h"
#include <entt/entity/registry.hpp>

namespace action::audio {
namespace {
template<typename T> T Default() { return {}; }
template<> AcousticMaterial Default<AcousticMaterial>() { return materials::acoustic::All.front(); }
template<> ContactSurface Default<ContactSurface>() { return WithPreset({}, surfaces::acoustic::Default); }

template<typename T> void Patch(entt::registry &r, entt::entity e, auto edit) {
    if (r.all_of<T>(e)) r.patch<T>(e, edit);
    else {
        auto value = Default<T>();
        edit(value);
        r.emplace<T>(e, std::move(value));
    }
}
} // namespace

void Apply(entt::registry &r, entt::entity, const Action &action) {
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
            [&](EnsureModalSettings) {
                const auto e = FindActiveEntity(r);
                if (!r.all_of<ModalSolveSettings>(e)) r.emplace<ModalSolveSettings>(e);
                if (!r.all_of<AcousticMaterial>(e)) r.emplace<AcousticMaterial>(e, Default<AcousticMaterial>());
                if (!r.all_of<ContactSurface>(e)) r.emplace<ContactSurface>(e, Default<ContactSurface>());
            },
            [&](const SetMaterialPreset &a) {
                if (const auto *material = materials::acoustic::Find(a.Name)) {
                    if (a.Striker) r.patch<Striker>(a.Entity, [&](auto &s) { s.Material = *material; });
                    else r.emplace_or_replace<AcousticMaterial>(a.Entity, *material);
                }
            },
            [&](const SetSurfacePreset &a) {
                for (const auto &preset : surfaces::acoustic::All)
                    if (a.Name == preset.Name) Patch<ContactSurface>(r, a.Entity, [&](auto &s) { s = WithPreset(std::move(s), preset); });
            },
            [&]<typename C, typename F, size_t N>(const PatchFields<C, F, N> &a) {
                Patch<C>(r, a.Entity, [&](C &c) {
                    for (size_t i = 0; i < N; ++i) {
                        assert(size_t(a.Offsets[i]) + sizeof(F) <= sizeof(C));
                        *reinterpret_cast<F *>(reinterpret_cast<std::byte *>(&c) + a.Offsets[i]) = a.Values[i];
                    }
                });
            },
            [&](const ApplyModalModel &a) { ::ApplyModalModel(r, a.SoundEntity, a.Path); },
            [&](const AssignVertexSamples &a) {
                auto frames = LoadAudioFrames(a.Path.string(), DeviceSampleRate(r));
                if (!frames.empty()) ::AssignVertexSample(r, FindActiveEntity(r), *a.MeshVertices, a.Path, std::move(frames));
            },
            [&](const ActivateRealImpactMicrophone &a) {
                const auto dir = r.get<const Path>(r.get<const Instance>(a.TargetSoundEntity).Entity).Value.parent_path();
                const auto &vertex_indices = r.get<const RealImpactVertices>(a.TargetSoundEntity).Vertices;
                const auto mic_index = r.get<const RealImpactMicrophone>(a.MicrophoneEntity).Index;
                auto samples = RealImpact::LoadSamples(dir, mic_index);
                if (!samples) {
                    r.ctx().get<Errors>().Messages.push_back(std::move(samples.error()));
                    return;
                }
                ::SetVertexSamples(r, a.TargetSoundEntity, vertex_indices, *samples);
                r.emplace_or_replace<RealImpactActiveMicrophone>(a.TargetSoundEntity, a.MicrophoneEntity);
            },
            [&](const RemoveVertexSamples &a) { ::RemoveVertexSamples(r, FindActiveEntity(r), a.MeshVertices); },
            [&]<typename T>(const Replace<T> &a) { r.emplace_or_replace<T>(a.Entity, a.Value); },
        },
        action
    );
}
} // namespace action::audio
