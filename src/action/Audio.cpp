#include "action/Audio.h"
#include "Path.h"
#include "action/Errors.h"
#include "audio/AudioSystem.h"
#include "audio/RealImpact.h"
#include "audio/SoundVertices.h"
#include "editor/AudioIntegration.h"
#include "project/Assets.h"
#include "project/Registry.h"
#include "render/Instance.h"
#include "scene/Entity.h"
#include <entt/entity/registry.hpp>

namespace action::audio {
namespace {
template<typename T> T Default() { return {}; }
template<> AcousticMaterial Default<AcousticMaterial>() { return materials::acoustic::All.front(); }
template<> ContactSurface Default<ContactSurface>() { return WithPreset({}, surfaces::acoustic::Default); }

template<typename T> void Patch(entt::registry &r, entt::entity e, auto edit) {
    if (r.all_of<T>(e)) project::Patch<T>(r, e, edit);
    else {
        auto value = Default<T>();
        edit(value);
        project::Emplace<T>(r, e, std::move(value));
    }
}
} // namespace

void Apply(entt::registry &r, entt::entity, const Action &action) {
    std::visit(
        overloaded{
            [&](const ApplyExciteImpact &a) {
                project::EmplaceOrReplace<MeshActiveElement>(r, r.get<Instance>(a.InstanceEntity).Entity, a.VertexIndex);
                project::EmplaceOrReplace<VertexForce>(r, a.InstanceEntity, a.VertexIndex, 1.f);
            },
            [&](ClearExciteImpacts) { project::Clear<VertexForce>(r); },
            [&](const SetModel &a) { ::SetModel(r, FindActiveEntity(r), a.Model); },
            [&](const SetExciteVertex &a) {
                const auto e = FindActiveEntity(r);
                project::Remove<VertexForce>(r, e);
                project::EmplaceOrReplace<MeshActiveElement>(r, GetActiveMeshEntity(r), a.MeshVertex);
                ::Stop(r, e);
            },
            [&](const StartExcite &a) {
                const auto e = FindActiveEntity(r);
                project::Remove<VertexForce>(r, e);
                project::Emplace<VertexForce>(r, e, a.Vertex, 1.f);
            },
            [&](StopExcite) { project::Remove<VertexForce>(r, FindActiveEntity(r)); },
            [&](DeleteSoundObject) { RemoveAudioComponents(r, FindActiveEntity(r)); },
            [&](const StartRecording &a) { project::EmplaceOrReplace<Recording>(r, FindActiveEntity(r), a.FrameCount); },
            [&](EnsureModalSettings) {
                const auto e = FindActiveEntity(r);
                if (!r.all_of<ModalSolveSettings>(e)) project::Emplace<ModalSolveSettings>(r, e);
                if (!r.all_of<AcousticMaterial>(e)) project::Emplace<AcousticMaterial>(r, e, Default<AcousticMaterial>());
                if (!r.all_of<ContactSurface>(e)) project::Emplace<ContactSurface>(r, e, Default<ContactSurface>());
            },
            [&](const SetMaterialPreset &a) {
                if (const auto *material = materials::acoustic::Find(a.Name)) {
                    if (a.Striker) project::Patch<Striker>(r, a.Entity, [&](auto &s) { s.Material = *material; });
                    else project::EmplaceOrReplace<AcousticMaterial>(r, a.Entity, *material);
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
                auto frames = LoadAudioFrames(project::ResolveAsset(r, a.Path).string(), DeviceSampleRate(r));
                if (!frames.empty()) ::AssignVertexSample(r, FindActiveEntity(r), *a.MeshVertices, a.Path, std::move(frames));
            },
            [&](const ActivateRealImpactMicrophone &a) {
                const auto &source = r.get<const RealImpactVertices>(a.TargetSoundEntity);
                if (!source.Samples.empty()) {
                    const auto mic_index = r.get<const RealImpactMicrophone>(a.MicrophoneEntity).Index;
                    auto samples = RealImpact::LoadSamples(r, source.Samples, mic_index);
                    if (!samples) {
                        r.ctx().get<Errors>().Messages.push_back(std::move(samples.error()));
                        return;
                    }
                    ::SetVertexSamples(r, a.TargetSoundEntity, source.Vertices, *samples);
                }
                project::EmplaceOrReplace<RealImpactActiveMicrophone>(r, a.TargetSoundEntity, a.MicrophoneEntity);
            },
            [&](const RemoveVertexSamples &a) { ::RemoveVertexSamples(r, FindActiveEntity(r), a.MeshVertices); },
            [&]<typename T>(const Replace<T> &a) { project::EmplaceOrReplace<T>(r, a.Entity, a.Value); },
        },
        action
    );
}
} // namespace action::audio
