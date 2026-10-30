#include "audio/AudioDevice.h"
#include "audio/AudioQueries.h"
#include "audio/ContactScene.h"
#include "scene/Entity.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/Viewport.h"
namespace fs = std::filesystem;
uint32_t DeviceSampleRate(const state::Registry &r) {
    const auto *info = r.Resources.find<const AudioDeviceInfo>();
    if (info && info->SampleRate) return info->SampleRate;
    // AUDIO_SAMPLE_RATE overrides the 48 kHz default when no device provides a rate.
    static const uint32_t fallback = [] {
        const char *env = std::getenv("AUDIO_SAMPLE_RATE");
        const auto rate = env ? std::atoi(env) : 0;
        return rate > 0 ? uint32_t(rate) : 48'000u;
    }();
    return fallback;
}

const std::vector<float> &GetSampleFrames(const state::Registry &r, const fs::path &path) {
    static const std::vector<float> EmptyFrames{};
    const auto &samples = r.Resources.get<const AudioSamples>().ByPath;
    const auto it = samples.find(path);
    return it != samples.end() ? it->second : EmptyFrames;
}
// Returns the active SoundVertices index derived from the mesh entity's MeshActiveElement.
// Returns zero when no active element is set.
uint32_t GetActiveVertexIndex(const state::Registry &r, state::Entity instance_entity) {
    const auto &excitable = r.get<const SoundVertices>(instance_entity);
    const auto mesh_entity = r.get<const Instance>(instance_entity).Entity;
    if (const auto *active = r.try_get<const MeshActiveElement>(mesh_entity)) {
        const auto &meshes = r.Resources.get<const MeshStore>();
        if (auto vi = FindSoundVertexIndex(meshes.Arenas().SoundVertices.Get(excitable.Vertices), active->Handle)) return *vi;
    }
    return 0;
}

// Returns the sample-store path assigned to the instance's active mesh vertex, if any.
std::optional<fs::path> ActiveSamplePath(const state::Registry &r, state::Entity instance_entity) {
    const auto *samples = r.try_get<const VertexSamples>(HierarchySource(r, instance_entity));
    if (!samples) return std::nullopt;
    const auto mesh_entity = r.get<const Instance>(instance_entity).Entity;
    const auto *active = r.try_get<const MeshActiveElement>(mesh_entity);
    if (!active) return std::nullopt;
    const auto it = samples->PathByVertex.find(active->Handle);
    return it != samples->PathByVertex.end() ? std::optional{it->second} : std::nullopt;
}

std::optional<vec3> ListenerPosition(const state::Registry &r) {
    const auto viewport = FindActiveViewport(r);
    if (viewport == state::Null) return std::nullopt;
    return r.get<const ViewCameraState>(viewport).Active.Position();
}

// Updates listener attenuation using inverse distance beyond ListenerDistance and a constant level within it.
void UpdateListenerGains(const state::Registry &r, ModalBank &b, vec3 listener_pos) {
    b.GainListener = listener_pos;
    for (uint32_t slot = 0; slot < uint32_t(b.Entities.size()); ++slot) {
        const auto e = b.Entities[slot];
        const auto *world = r.valid(e) ? WorldTransformOf(r, e) : nullptr;
        const float distance = world ? Distance(listener_pos, world->P) : ListenerDistance;
        std::atomic_ref{b.ListenerGain[slot]}.store(ListenerDistance / std::max(distance, ListenerDistance), std::memory_order_relaxed);
    }
}

// The device-to-main-thread ring overwrites old audio instead of blocking the callback.
struct MasterCapture {
    std::vector<float> Ring;
    std::atomic<uint64_t> Written{0}, Read{0};
};

uint32_t BeginAudioCapture(state::Registry &r) {
    const auto *info = r.Resources.find<const AudioDeviceInfo>();
    const auto rate = info ? info->SampleRate : 0u;
    if (rate == 0) return 0;
    auto &capture = r.Resources.emplace<MasterCapture>();
    capture.Ring.assign(size_t(rate) * 2, 0.f);
    capture.Written.store(0, std::memory_order_relaxed);
    capture.Read.store(0, std::memory_order_relaxed);
    return rate;
}

void EndAudioCapture(state::Registry &r) { r.Resources.erase<MasterCapture>(); }

void RenderAudioOffline(state::Registry &r, std::vector<float> &out, uint32_t frame_count) {
    const auto first = out.size();
    out.resize(first + frame_count);
    ProcessAudio(r, out.data() + first, frame_count);
}

void DrainAudioCapture(state::Registry &r, std::vector<float> &out) {
    auto *capture = r.Resources.find<MasterCapture>();
    if (!capture || capture->Ring.empty()) return;
    const auto written = capture->Written.load(std::memory_order_acquire);
    auto read = capture->Read.load(std::memory_order_relaxed);
    // Resume from the oldest retained frame after an overwrite.
    const auto capacity = uint64_t(capture->Ring.size());
    if (written - read > capacity) read = written - capacity;
    for (auto pos = size_t(read % capacity); read < written; ++read) {
        out.push_back(capture->Ring[pos]);
        if (++pos == capacity) pos = 0;
    }
    capture->Read.store(read, std::memory_order_relaxed);
}

// The pressure a full-scale device sample represents, Pa: the monitor level, 120 dB SPL, the threshold of pain.
// Maps the monitor pressure ceiling to full scale.
constexpr float FullScalePressure{20.f};

// Uses instant attack and 100 ms release while preserving relative levels below the peak envelope.
void MonitorFrames(state::Registry &r, std::span<float> frames, MonitorLimiter &limiter) {
    const float release = std::exp(-1.f / (0.1f * float(DeviceSampleRate(r))));
    auto envelope = limiter.Envelope;
    for (auto &frame : frames) {
        const float x = frame / FullScalePressure;
        envelope = std::max(std::abs(x), envelope * release);
        frame = envelope > 1.f ? x / envelope : x;
    }
    limiter.Envelope = envelope;
}

void ProcessAudio(state::Registry &r, float *output, uint32_t frame_count, bool monitor) {
    std::fill_n(output, frame_count, 0.f);
    auto &m = r.Resources.get<ModalAudio>();
    // The mix is pressure at the view camera.
    // The device path runs this on the audio thread, which cannot read the registry, so its gains are written by the frame handler.
    // Offline rendering uses its viewport camera on the main thread.
    if (!monitor) {
        if (const auto listener = ListenerPosition(r)) UpdateListenerGains(r, LiveBank(m), *listener);
    }
    RenderModal(m, output, frame_count);

    const auto *controls = r.try_get<const ModalSoundControls>(state::Document);
    // Recorded samples are normalized, so they enter the pressure mix at the monitor calibration and a full-scale sample plays at full scale.
    const float sample_gain = (controls ? controls->SampleGain : ModalSoundControls{}.SampleGain) * FullScalePressure;
    for (const auto [entity, model] : r.view<const SoundVerticesModel>().each()) {
        if (model == SoundVerticesModel::Samples) {
            auto *samples = r.try_edit<SamplePlayback>(entity);
            if (!samples || samples->Stopped) continue;
            const auto path = ActiveSamplePath(r, entity);
            if (!path) continue;
            const auto &impact_samples = GetSampleFrames(r, *path);
            for (uint32_t i = 0; i < frame_count; ++i) {
                output[i] += (samples->Frame < impact_samples.size() ? impact_samples[samples->Frame++] : 0.0f) * sample_gain;
            }
        } else if (model == SoundVerticesModel::Modal) {
            if (auto *recording = r.try_edit<Recording>(entity)) {
                for (uint32_t i = 0; i < frame_count && !recording->Complete(); ++i) recording->Record(output[i]);
            }
        }
    }

    if (auto *capture = r.Resources.find<MasterCapture>(); capture && !capture->Ring.empty()) {
        const auto capacity = uint64_t(capture->Ring.size());
        auto written = capture->Written.load(std::memory_order_relaxed);
        auto pos = size_t(written % capacity);
        for (uint32_t i = 0; i < frame_count; ++i, ++written) {
            capture->Ring[pos] = output[i];
            if (++pos == capacity) pos = 0;
        }
        capture->Written.store(written, std::memory_order_release);
    }

    // The monitor stage, after the capture tap: device units at the monitor level.
    if (monitor) MonitorFrames(r, {output, frame_count}, r.Resources.get<MonitorLimiter>());
}
