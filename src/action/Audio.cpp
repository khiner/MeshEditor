#include "action/Audio.h"
#include "Variant.h"
#include "action/Errors.h"
#include "audio/AudioOutputConfig.h"
#include "audio/RealImpact.h"
#include "audio/SoundVertices.h"
#include "editor/AudioIntegration.h"
#include "project/Assets.h"
#include "render/Instance.h"
#include "scene/Entity.h"
#include "scene/HierarchyInstance.h"

namespace action::audio {
namespace {
template<typename T> T Default() { return {}; }
template<> AcousticMaterial Default<AcousticMaterial>() { return materials::acoustic::All.front(); }
template<> ContactSurface Default<ContactSurface>() { return WithPreset({}, surfaces::acoustic::Default); }

template<typename T> void Patch(state::Registry &r, state::Entity e, auto edit) {
    if (r.all_of<T>(e)) r.patch<T>(e, edit);
    else {
        auto value = Default<T>();
        edit(value);
        r.emplace<T>(e, std::move(value));
    }
}
} // namespace

void Apply(state::Registry &r, const Action &action) {
    const auto target = [&](state::Entity entity) { return entity == state::Null ? FindActiveEntity(r) : entity; };
    const auto definition = [&](state::Entity e) { return HierarchySource(r, target(e)); };
    std::visit(
        overloaded{
            [&](const ApplyExciteImpact &a) {
                r.emplace_or_replace<MeshActiveElement>(r.get<Instance>(a.InstanceEntity).Entity, a.VertexIndex);
                r.emplace_or_replace<VertexForce>(a.InstanceEntity, a.VertexIndex, 1.f);
            },
            [&](ClearExciteImpacts) { r.clear<VertexForce>(); },
            [&](const SetModel &a) { ::SetModel(r, definition(a.Entity), a.Model); },
            [&](const SetExciteVertex &a) {
                const auto e = target(a.Entity);
                r.remove<VertexForce>(e);
                r.emplace_or_replace<MeshActiveElement>(GetMeshEntity(r, e), a.MeshVertex);
                ::Stop(r, e);
            },
            [&](const StartExcite &a) {
                const auto e = target(a.Entity);
                r.remove<VertexForce>(e);
                r.emplace<VertexForce>(e, a.Vertex, 1.f);
            },
            [&](const StopExcite &a) { r.remove<VertexForce>(target(a.Entity)); },
            [&](const DeleteSoundObject &a) { RemoveAudioComponents(r, definition(a.Entity)); },
            [&](const StartRecording &a) { r.emplace_or_replace<Recording>(target(a.Entity), a.FrameCount); },
            [&](const EnsureModalSettings &a) {
                const auto e = definition(a.Entity);
                if (!r.all_of<ModalSolveSettings>(e)) r.emplace<ModalSolveSettings>(e);
                if (!r.all_of<AcousticMaterial>(e)) r.emplace<AcousticMaterial>(e, Default<AcousticMaterial>());
                if (!r.all_of<ContactSurface>(e)) r.emplace<ContactSurface>(e, Default<ContactSurface>());
            },
            [&](const SetMaterialPreset &a) {
                if (const auto *material = materials::acoustic::Find(a.Name)) {
                    if (a.Striker) r.patch<Striker>(state::Document, [&](auto &s) { s.Material = *material; });
                    else r.emplace_or_replace<AcousticMaterial>(definition(a.Entity), *material);
                }
            },
            [&](const SetSurfacePreset &a) {
                for (const auto &preset : surfaces::acoustic::All) {
                    if (a.Name == preset.Name) Patch<ContactSurface>(r, definition(a.Entity), [&](auto &s) { s = WithPreset(std::move(s), preset); });
                }
            },
            [&]<typename C, typename F>(const PatchFields<C, F> &a) {
                assert(size_t(a.Offset) + sizeof(F) <= sizeof(C));
                Patch<C>(r, definition(a.Entity), [&](C &c) { *reinterpret_cast<F *>(reinterpret_cast<std::byte *>(&c) + a.Offset) = a.Value; });
            },
            [&](const ApplyModalModel &a) { ::ApplyModalModel(r, definition(a.SoundEntity), a.Path); },
            [&](const AssignVertexSamples &a) {
                const auto entity = definition(a.Entity);
                if (!r.all_of<Instance>(entity)) return Fail(r, "The sound object no longer exists");
                auto frames = LoadAudioFrames(project::ResolveAsset(r, a.Path), DeviceSampleRate(r));
                if (!frames.empty()) ::AssignVertexSample(r, entity, *a.MeshVertices, a.Path, std::move(frames));
            },
            [&](const ActivateRealImpactMicrophone &a) {
                const auto microphone = definition(a.Entity);
                const auto sound = HierarchySource(r, a.TargetSoundEntity);
                const auto &source = r.get<const RealImpactVertices>(sound);
                if (!source.Samples.empty()) {
                    const auto mic_index = r.get<const RealImpactMicrophone>(microphone).Index;
                    auto samples = RealImpact::LoadSamples(r, source.Samples, mic_index);
                    if (!samples) {
                        Fail(r, std::move(samples.error()));
                        return;
                    }
                    ::SetVertexSamples(r, sound, source.Vertices, *samples);
                }
                r.emplace_or_replace<RealImpactActiveMicrophone>(sound, microphone);
            },
            [&](const RemoveVertexSamples &a) { ::RemoveVertexSamples(r, definition(a.Entity), a.MeshVertices); },
            [&](const SetOutputDevice &a) { r.Resources.get<AudioOutputConfig>() = {a.DeviceName, a.SampleRate}; },
        },
        action
    );
}
} // namespace action::audio
