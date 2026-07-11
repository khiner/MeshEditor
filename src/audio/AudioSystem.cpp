#include "AudioSystem.h"
#include "AudioDevice.h"
#include "FFTData.h"
#include "Job.h"
#include "ModalAudio.h"
#include "Reactive.h"
#include "action/ActionApply.h"
#include "action/Audio.h"
#include "audio/SoundVertices.h"
#include "mesh/Mesh.h"
#include "mesh/Tets.h"
#include "render/Instance.h"
#include "scene/WorldTransform.h"
#include "selection/SelectionBitset.h"
#include "ui/FieldEdit.h"
#include "viewport/InteractionComponents.h"

#include "ContactModel.h"
#include "ModalModelFile.h"
#include "ModalWarmStart.h"
#include "implot.h"
#include "imspinner.h"
#include "mesh2modes.h"
#include "miniaudio.h"
#include "tetgen.h" // Needed for `unique_ptr<tetgenio>` dereference.

#include "ui/HelpMarker.h" // depends on imgui

#include <nfd.h>

#include <iostream>
#include <numbers>

namespace fs = std::filesystem;

// Ranges cover the acoustic material presets with headroom (see materials::acoustic::All).
// Poisson's ratio stays below 0.5 (avoid divide-by-zero). Beta's floor is an effective zero that
// keeps its whole range above the logarithmic slider's zero epsilon.
template<> struct FieldLimits<&AcousticMaterial::Properties, &AcousticMaterialProperties::Density> : Within<1., 25000.> {};
template<> struct FieldLimits<&AcousticMaterial::Properties, &AcousticMaterialProperties::YoungModulus> : Within<1e5, 1e12> {};
template<> struct FieldLimits<&AcousticMaterial::Properties, &AcousticMaterialProperties::PoissonRatio> : Within<0., 0.49> {};
template<> struct FieldLimits<&AcousticMaterial::Properties, &AcousticMaterialProperties::Alpha> : Within<0., 200.> {};
template<> struct FieldLimits<&AcousticMaterial::Properties, &AcousticMaterialProperties::Beta> : Within<1e-9, 1e-4> {};
template<> struct FieldLimits<&ModalSolveSettings::SolveResolution> : Within<0.05, 1.> {};

// Striker capsule dimensions, in meters.
template<> struct FieldLimits<&Striker::TipRadius> : Within<0.0005, 0.1> {};
template<> struct FieldLimits<&Striker::Length> : Within<0.001, 1.> {};

// Modal synthesis controls.
template<> struct FieldLimits<&ModalGain::Value> : Within<0., 2.> {};
template<> struct FieldLimits<&ModalTuning::FundamentalFreq> : Within<20., 16000.> {};
template<> struct FieldLimits<&ModalTuning::T60Scale> : Within<0.1, 10.> {};
template<> struct FieldLimits<&ModalSoundControls::ClickGain> : Within<0., 10.> {};

using std::ranges::iota_view, std::ranges::nth_element, std::ranges::max_element, std::ranges::to;
using std::views::transform;

static constexpr uint32_t SampleRate = 48'000; // todo respect device sample rate

namespace {
// Per-sound-object component. Maps mesh vertex handles to sample keys in the scene-level AudioSamples store.
// Only vertices that have a sample appear in the map.
struct VertexSamples {
    std::map<uint32_t, fs::path> PathByVertex;
    uint32_t Frame{0};
    bool Stopped{true};

    std::optional<fs::path> FindPath(uint32_t mesh_vertex) const {
        auto it = PathByVertex.find(mesh_vertex);
        return it != PathByVertex.end() ? std::optional{it->second} : std::nullopt;
    }
    void Stop() { Stopped = true; }
    void Play() {
        Frame = 0;
        Stopped = false;
    }
};

// Scene-singleton audio sample store. Keyed by `fs::path` so a single loaded sample shared by
// many vertices (or many sound objects) is stored exactly once. Entries are refcounted and
// erased when the last reference drops.
struct AudioSamples {
    struct Entry {
        std::vector<float> Frames;
        uint32_t RefCount{0};
    };
    std::unordered_map<fs::path, Entry> ByPath;
};

const std::vector<float> &GetSampleFrames(const entt::registry &r, entt::entity viewport, const fs::path &path) {
    static const std::vector<float> EmptyFrames{};
    if (path.empty()) return EmptyFrames;
    const auto *store = r.try_get<const AudioSamples>(viewport);
    if (!store) return EmptyFrames;
    const auto it = store->ByPath.find(path);
    return it != store->ByPath.end() ? it->second.Frames : EmptyFrames;
}

// Inserts frames if `path` is new, otherwise reuses existing frames. Bumps refcount either way.
void AcquireSample(entt::registry &r, entt::entity viewport, const fs::path &path, std::vector<float> &&frames) {
    if (path.empty()) return;
    auto &store = r.get_or_emplace<AudioSamples>(viewport);
    auto [it, inserted] = store.ByPath.try_emplace(path);
    if (inserted) it->second.Frames = std::move(frames);
    ++it->second.RefCount;
}

// Decrements refcount; erases the entry (and the component if empty) when it hits 0.
void ReleaseSample(entt::registry &r, entt::entity viewport, const fs::path &path) {
    if (path.empty()) return;
    auto *store = r.try_get<AudioSamples>(viewport);
    if (!store) return;
    const auto it = store->ByPath.find(path);
    if (it == store->ByPath.end()) return;
    if (--it->second.RefCount == 0) store->ByPath.erase(it);
    if (store->ByPath.empty()) r.remove<AudioSamples>(viewport);
}
} // namespace

std::vector<float> LoadAudioFrames(const std::string &file_path) {
    const ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 1, SampleRate);
    ma_decoder decoder;
    if (ma_decoder_init_file(file_path.c_str(), &config, &decoder) != MA_SUCCESS) {
        std::cerr << std::format("Failed to open audio file: {}\n", file_path);
        return {};
    }
    ma_uint64 total_frames = 0;
    if (ma_decoder_get_length_in_pcm_frames(&decoder, &total_frames) != MA_SUCCESS || total_frames == 0) {
        ma_decoder_uninit(&decoder);
        std::cerr << std::format("Failed to read length of audio file: {}\n", file_path);
        return {};
    }
    std::vector<float> frames(total_frames);
    ma_uint64 frames_read = 0;
    ma_decoder_read_pcm_frames(&decoder, frames.data(), total_frames, &frames_read);
    frames.resize(frames_read);
    ma_decoder_uninit(&decoder);
    return frames;
}

void AssignVertexSample(
    entt::registry &r, entt::entity viewport, entt::entity e,
    std::span<const uint32_t> mesh_vertices, fs::path path, std::vector<float> &&frames
) {
    if (mesh_vertices.empty() || path.empty()) return;
    auto &vs = r.get_or_emplace<VertexSamples>(e);
    vs.Stop();

    // `frames` is consumed on the first new-path insertion; subsequent AcquireSample calls
    // see the path already in the store and only bump refcount, so the moved-from vector is ignored.
    bool vs_changed = false;
    for (const uint32_t mv : mesh_vertices) {
        auto [it, inserted] = vs.PathByVertex.try_emplace(mv, path);
        if (!inserted) {
            if (it->second == path) continue;
            ReleaseSample(r, viewport, it->second);
            it->second = path;
        }
        AcquireSample(r, viewport, path, std::move(frames)); // NOLINT(bugprone-use-after-move) only the first new-path call reads frames
        vs_changed = true;
    }
    if (vs_changed) r.patch<VertexSamples>(e, [](auto &) {});
    if (!r.all_of<SoundVerticesModel>(e)) r.emplace<SoundVerticesModel>(e, SoundVerticesModel::Samples);
}

void RemoveVertexSamples(
    entt::registry &r, entt::entity viewport, entt::entity e,
    std::span<const uint32_t> mesh_vertices
) {
    auto *vs = r.try_get<VertexSamples>(e);
    if (!vs || mesh_vertices.empty()) return;
    vs->Stop();
    bool vs_changed = false;
    for (const uint32_t mv : mesh_vertices) {
        const auto it = vs->PathByVertex.find(mv);
        if (it == vs->PathByVertex.end()) continue;
        ReleaseSample(r, viewport, it->second);
        vs->PathByVertex.erase(it);
        vs_changed = true;
    }
    if (vs_changed) r.patch<VertexSamples>(e, [](auto &) {});
    if (vs->PathByVertex.empty()) {
        if (r.all_of<ModalModes>(e)) r.remove<VertexSamples>(e);
        else RemoveAudioComponents(r, e);
    }
}

void SetVertexSamples(
    entt::registry &r, entt::entity viewport, entt::entity e,
    std::span<const uint32_t> mesh_vertices, std::vector<LoadedSample> &&samples
) {
    for (size_t i = 0; i < samples.size() && i < mesh_vertices.size(); ++i) {
        AssignVertexSample(r, viewport, e, {&mesh_vertices[i], 1}, std::move(samples[i].first), std::move(samples[i].second));
    }
}

namespace {
// Returns the index (into SoundVertices::Vertices) of the active vertex for this instance entity,
// derived from MeshActiveElement on the mesh entity. Returns 0 if no active element is set.
uint32_t GetActiveVertexIndex(const entt::registry &r, entt::entity instance_entity) {
    const auto &excitable = r.get<const SoundVertices>(instance_entity);
    const auto mesh_entity = r.get<const Instance>(instance_entity).Entity;
    if (const auto *active = r.try_get<const MeshActiveElement>(mesh_entity)) {
        if (auto vi = excitable.FindVertexIndex(active->Handle)) return *vi;
    }
    return 0;
}

// Returns the sample-store path assigned to the instance's active mesh vertex, if any.
std::optional<fs::path> ActiveSamplePath(const entt::registry &r, entt::entity instance_entity) {
    const auto *samples = r.try_get<const VertexSamples>(instance_entity);
    if (!samples) return std::nullopt;
    const auto mesh_entity = r.get<const Instance>(instance_entity).Entity;
    const auto *active = r.try_get<const MeshActiveElement>(mesh_entity);
    return active ? samples->FindPath(active->Handle) : std::nullopt;
}

/***** Modal synthesis bank *****/

// Base output level of a modal object. ModalGain scales relative to this.
constexpr float ModalLevel{0.14f};

// Reduce a (possibly non-uniform or mirrored) world scale to a positive size ratio relative to the baked size.
float UniformScaleRatio(const entt::registry &r, entt::entity e, const ModalModes &modes) {
    const auto size = [](vec3 v) { const auto a = glm::abs(v); return (a.x + a.y + a.z) / 3.f; };
    const auto *world = r.try_get<const WorldTransform>(e);
    const float baked = size(modes.BakedScale);
    return world && baked > 0 ? std::clamp(size(world->S) / baked, 0.001f, 1000.f) : 1.f;
}

// Recompute an object's resonator coefficients and output gain.
// Frequencies shift proportionally with the fundamental target and inversely with size.
// Uniform scaling sends omega -> omega/scale, so when Rayleigh alpha (on the mesh entity) is known,
// d = (alpha + beta*omega^2)/2 becomes d' = alpha/2 + (d - alpha/2)/scale^2, then T60 = ln(1000)/d'.
// (T60 == 0 is the undamped sentinel and stays 0, muting the mode.)
void RetuneModalObject(const entt::registry &r, ModalAudio &m, uint32_t slot, entt::entity e) {
    const auto &modes = r.get<const ModalModes>(e);
    const auto mode_count = modes.Freqs.size();
    if (mode_count == 0) return;

    const float scale = UniformScaleRatio(r, e, modes);
    const auto *tuning = r.try_get<const ModalTuning>(e);
    const float fundamental = tuning ? tuning->FundamentalFreq : 0.f;
    const float freq_ratio = (fundamental > 0 && modes.Freqs.front() > 0 ? fundamental / modes.Freqs.front() : 1.f) / scale;
    const float t60_scale = tuning ? tuning->T60Scale : 1.f;
    std::optional<double> alpha;
    if (const auto *inst = r.try_get<const Instance>(e)) {
        if (const auto *mat = r.try_get<const AcousticMaterial>(inst->Entity)) alpha = mat->Properties.Alpha;
    }

    static constexpr float Ln1000 = 6.907755279f;
    std::vector<float> freqs(mode_count), t60s(mode_count);
    for (size_t k = 0; k < mode_count; ++k) {
        freqs[k] = modes.Freqs[k] * freq_ratio;
        const float t60 = modes.T60s[k];
        if (t60 <= 0) {
            t60s[k] = 0;
            continue;
        }
        float d = Ln1000 / t60;
        if (alpha) d = float(*alpha / 2) + (d - float(*alpha / 2)) / (scale * scale);
        t60s[k] = t60_scale * Ln1000 / std::max(d, 1e-9f);
    }
    TuneModalObject(m, slot, freqs, t60s);

    // The scale^(-3/2) is the mass-normalization amplitude law: mass grows as scale^3, so a larger object rings quieter per impulse.
    const auto *gain = r.try_get<const ModalGain>(e);
    m.OutGain[slot] = ModalLevel * (gain ? gain->Value : 1.f) * std::pow(scale, -1.5f) / float(mode_count);
}

// Rebuild the bank from every modal sound object. Structural, so it briefly excludes the audio thread.
void RebuildModalBank(entt::registry &r) {
    auto &m = r.ctx().get<ModalAudio>();
    std::scoped_lock lock{m.StructureMutex};
    ClearModalObjects(m);
    const auto *res = r.ctx().find<AudioDeviceResource>();
    m.SampleRate = res && res->SampleRate ? float(res->SampleRate) : 48'000.f;
    for (auto e : r.view<const ModalModes, const SoundVertices>()) {
        const auto &modes = r.get<const ModalModes>(e);
        if (modes.Freqs.empty()) continue;
        const auto slot = AddModalObject(m, e, modes);
        RetuneModalObject(r, m, slot, e);
    }
}

} // namespace

/***** Free functions for sound object control *****/

void Stop(entt::registry &r, entt::entity e) {
    if (auto *samples = r.try_get<VertexSamples>(e)) samples->Stop();
    if (r.all_of<ModalModes>(e)) {
        auto &m = r.ctx().get<ModalAudio>();
        if (auto slot = FindModalObject(m, e)) EnqueueModalEvent(m, {.Kind = ModalEventKind::Silence, .Object = *slot});
    }
}

void SetModel(entt::registry &r, entt::entity e, SoundVerticesModel model) {
    Stop(r, e);

    const bool is_sample = model == SoundVerticesModel::Samples && r.all_of<VertexSamples>(e);
    const bool is_modal = model == SoundVerticesModel::Modal && r.all_of<ModalModes>(e);
    if (!is_sample && !is_modal) return;

    r.emplace_or_replace<SoundVerticesModel>(e, model);
}

namespace {
// Strike impact angle relative to the surface, as a joystick position in the unit disk. Center strikes
// along the surface normal, the rim tilts the impulse 90 degrees into the tangent plane. UI-only.
vec2 ImpulseAngle{0, 0};

// Unit surface normal at a mesh vertex.
vec3 VertexNormal(const Mesh &mesh, uint32_t vertex) { return glm::normalize(mesh.GetNormal(Mesh::VH{vertex})); }

// Tilt a unit normal toward the surface by a joystick position in the unit disk (center leaves it along n).
vec3 TiltAlongNormal(vec3 n, vec2 joy) {
    const float r = glm::length(joy);
    if (r < 1e-6f) return n;
    // Orthonormal tangent basis from the normal (Duff et al. 2017).
    const float s = n.z >= 0 ? 1.f : -1.f;
    const float a = -1.f / (s + n.z);
    const float b = n.x * n.y * a;
    const vec3 t{1.f + s * n.x * n.x * a, s * b, -s * n.x};
    const vec3 bt{b, s + n.y * n.y * a, -n.y};
    const float theta = std::min(r, 1.f) * 1.57079633f; // radius maps to [0, pi/2]
    return std::cos(theta) * n + std::sin(theta) * (joy.x * t + joy.y * bt) / r;
}

// Strike direction: the excited vertex's normal, tilted by the current impact angle.
vec3 ExciteDirection(const entt::registry &r, entt::entity e, uint32_t vertex) {
    const auto n = VertexNormal(GetMesh(r, r.get<const Instance>(e).Entity), vertex);
    return TiltAlongNormal(n, ImpulseAngle);
}

// Estimate the strike's contact parameters from the Hertz model and enqueue the impact.
// The half-sine contact-force pulse of duration tau has unit sample sum, so its spectrum is flat
// at DC and rolls off above ~1/tau: shorter contact is brighter.
// Impulse magnitude rides in the mode excitation gains, not the pulse.
void TriggerModalStrike(entt::registry &r, entt::entity e, uint32_t excitable_index, float force, float contact_speed) {
    auto &m = r.ctx().get<ModalAudio>();
    const auto slot = FindModalObject(m, e);
    if (!slot) return;

    const auto &modes = r.get<const ModalModes>(e);
    if (excitable_index >= modes.Vertices.size()) return;
    const vec3 dir = ExciteDirection(r, e, modes.Vertices[excitable_index]);

    const auto *cd = r.try_get<const ContactDynamics>(e);
    const auto *mat = r.try_get<const AcousticMaterial>(r.get<const Instance>(e).Entity);
    // A short default contact with no click applies when the material or contact dynamics are missing.
    double tau = 1e-4; // seconds
    double accel_amp = 0;
    if (cd && mat) {
        const auto *device = r.ctx().find<AudioDeviceResource>();
        const auto *striker_ptr = device ? r.try_get<const Striker>(device->Viewport) : nullptr;
        const Striker striker = striker_ptr ? *striker_ptr : Striker{};
        // Contact time scales linearly with the object's current size.
        tau = EstimateContactTime(*cd, excitable_index, dir, contact_speed, mat->Properties, striker, UniformScaleRatio(r, e, modes));
        // Per-strike acceleration-noise amplitude: the impulse magnitude scaled down by material density.
        accel_amp = ReducedContactMass(*cd, excitable_index, dir, striker) * std::abs(double(contact_speed)) / mat->Properties.Density;
    }
    const float step = float(1.0 / (tau * m.SampleRate));
    EnqueueModalEvent(
        m,
        {
            .Kind = ModalEventKind::Impact,
            .Object = *slot,
            .ExPos = excitable_index,
            .Jx = dir.x * force,
            .Jy = dir.y * force,
            .Jz = dir.z * force,
            .PulseStep = step,
            .PulseGamma = std::numbers::pi_v<float> / 2 * step,
            .AccelAmp = float(accel_amp),
        }
    );
}

// Survives the frame-end clear, so the audio handler picks up world-transform changes made after it already ran, on the following frame.
struct ModalScaleTracker {
    entt::storage_for_t<entt::reactive> Storage;
    void Bind(entt::registry &r) {
        Storage.bind(r);
        Storage.on_update<WorldTransform>();
    }
};

// Reactive change types for audio system
namespace audio_changes {
struct VertexForce {};
struct ModalModes {};
struct ModalParams {};
struct RecordingStart {};
struct SoundVerticesDerivation {};
struct ContactDynamicsDerivation {};
struct AcousticMaterialEdit {};
struct AudioConfig {};
struct AudioMix {};
} // namespace audio_changes

/***** Impact spectrum analysis *****/

constexpr void ApplyCosineWindow(float *w, uint32_t n, const float *coeff, uint32_t ncoeff) {
    if (n == 1) {
        w[0] = 1.0;
        return;
    }

    const uint32_t wlength = n;
    for (uint32_t i = 0; i < n; ++i) {
        float wi = 0.0;
        for (uint32_t j = 0; j < ncoeff; ++j) wi += coeff[j] * __cospi(float(2 * i * j) / float(wlength));
        w[i] = wi;
    }
}

// Create Blackman-Harris window
constexpr std::vector<float> CreateBlackmanHarris(uint32_t n) {
    std::vector<float> window(n);
    static constexpr float coeff[4] = {0.35875, -0.48829, 0.14128, -0.01168};
    ApplyCosineWindow(window.data(), n, coeff, sizeof(coeff) / sizeof(float));
    return window;
}

constexpr std::vector<float> ApplyWindow(const std::vector<float> &window, const float *data) {
    std::vector<float> windowed(window.size());
    for (uint32_t i = 0; i < window.size(); ++i) windowed[i] = window[i] * data[i];
    return windowed;
}

std::optional<float> EstimateFundamentalFrequency(const FFTData &fft) {
    const auto *data = fft.Complex;
    const size_t n_bins = fft.NumReal / 2 + 1;

    std::vector<float> mag_db(n_bins);
    for (size_t i = 0; i < n_bins; ++i) {
        const auto mag_sq = data[i][0] * data[i][0] + data[i][1] * data[i][1];
        mag_db[i] = 10.f * std::log10f(std::max(mag_sq, 1e-20f));
    }

    // Noise floor from upper half median
    std::vector<float> upper(mag_db.begin() + n_bins / 2, mag_db.end());
    nth_element(upper, upper.begin() + upper.size() / 2);
    const float threshold = upper[upper.size() / 2] + 15.f;

    constexpr size_t W{15}; // Prominence window
    const size_t min_bin = 50 * fft.NumReal / SampleRate;
    for (size_t i = std::max(min_bin, W); i < n_bins - W; ++i) {
        // Local maximum?
        if (mag_db[i] <= mag_db[i - 1] || mag_db[i] <= mag_db[i + 1] || mag_db[i] < threshold) continue;

        constexpr float ProminenceThresholdDb{10.f};
        // Prominence check: peak must be above the local mean by ProminenceThresholdDb
        float local_sum = 0;
        for (size_t j = i - W; j <= i + W; ++j) local_sum += mag_db[j];
        const float local_mean = local_sum / (2 * W + 1);
        if (mag_db[i] - local_mean >= ProminenceThresholdDb) return i * SampleRate / fft.NumReal;
    }
    return std::nullopt;
}

// Capture a short audio segment shortly after the impact for FFT.
FFTData ComputeFft(const std::vector<float> &frames) {
    static constexpr uint32_t FftStartFrame = 30, FftEndFrame = SampleRate / 16;
    static const auto BHWindow = CreateBlackmanHarris(FftEndFrame - FftStartFrame);
    return {ApplyWindow(BHWindow, frames.data() + FftStartFrame)};
}

/***** Modal solve jobs *****/

struct ModalGenerationResult {
    std::filesystem::path ModelPath; // Result file, relative to ModalModelsDir(). Empty when the solve failed or was cancelled
    std::shared_ptr<const Eigen::MatrixXf> Basis; // Full eigenvector basis, seeding the next solve
    size_t TetInputsHash;
};

// An in-flight modal solve, at most one per sound entity.
struct ModalSolveJob {
    entt::entity Entity, Viewport;
    Job<ModalGenerationResult> Work;
};
struct ModalSolveJobs {
    std::vector<std::shared_ptr<ModalSolveJob>> Jobs;
};

bool IsSolving(const entt::registry &r, entt::entity e) {
    return std::ranges::any_of(r.ctx().get<const ModalSolveJobs>().Jobs, [e](const auto &job) { return job->Entity == e; });
}

// The cancelled job thread exits at its next checkpoint, and its result is discarded on arrival.
void CancelModalSolves(entt::registry &r, entt::entity e) {
    for (auto &job : r.ctx().get<ModalSolveJobs>().Jobs) {
        if (job->Entity == e) job->Work.Monitor->RequestCancel();
    }
}

/***** Modal model derivation *****/

// Exact re-derivation of `modes` for `props` (see modal::RescaleModes). A fundamental pinned at
// solve time (e.g. matched to a recording) stays pinned. Empty when the edit is not exactly
// scalable (Poisson ratio differs).
std::optional<ModalModes> RescaledModes(const ModalEigenSummary &summary, const ModalModes &modes, const AcousticMaterialProperties &props) {
    std::optional<float> fundamental;
    if (!modes.Freqs.empty() && modes.OriginalFundamentalFreq > 0 && modes.Freqs.front() != modes.OriginalFundamentalFreq) fundamental = modes.Freqs.front();
    return modal::RescaleModes(summary, modes, props, {.FundamentalFreq = fundamental});
}

// A synth tuning still at its default (fundamental == the old model's lowest mode) follows the
// new model, while a user-set tuning stays pinned.
// Intentional registry writes outside Apply: the model and tuning are derived state.
void ReplaceModalModes(entt::registry &r, entt::entity e, ModalModes new_modes) {
    const auto *tuning = r.try_get<const ModalTuning>(e);
    const auto *old_modes = r.try_get<const ModalModes>(e);
    if (tuning && old_modes && !old_modes->Freqs.empty() && !new_modes.Freqs.empty() &&
        tuning->FundamentalFreq == old_modes->Freqs.front() && tuning->FundamentalFreq != new_modes.Freqs.front()) {
        r.replace<ModalTuning>(e, ModalTuning{new_modes.Freqs.front(), tuning->T60Scale});
    }
    r.emplace_or_replace<ModalModes>(e, std::move(new_modes));
}

// Re-derive the entity's modal model for the mesh's current material. A Poisson-ratio change is
// not scalable and leaves the model as-is until a re-solve.
void RescaleModalObject(entt::registry &r, entt::entity e, const AcousticMaterialProperties &props) {
    const auto &modes = r.get<const ModalModes>(e);
    auto rescaled = RescaledModes(r.get<const ModalEigenSummary>(e), modes, props);
    if (rescaled && *rescaled != modes) ReplaceModalModes(r, e, std::move(*rescaled));
}

/***** Modal solve inputs *****/

// Identifies the tet-mesh inputs of a modal solve. Identical inputs produce identical tet
// topology, so a basis solved over them can warm-start the next solve (e.g. a material edit).
size_t HashTetInputs(const std::vector<vec3> &positions, const std::vector<uint32_t> &triangle_indices, TetGenOptions opts) {
    const auto bytes = [](const auto &v) { return std::string_view{reinterpret_cast<const char *>(v.data()), v.size() * sizeof(v[0])}; };
    const std::hash<std::string_view> hash;
    size_t seed = hash(bytes(positions));
    const auto combine = [&seed](size_t v) { seed ^= v + 0x9e3779b97f4a7c15 + (seed << 6) + (seed >> 2); };
    combine(hash(bytes(triangle_indices)));
    combine(std::hash<bool>{}(opts.PreserveSurface));
    combine(std::hash<bool>{}(opts.Quality));
    combine(std::hash<float>{}(opts.SimplifyRatio));
    return seed;
}

// The excitation vertices a solve would use: the existing excitable vertices when copying,
// else evenly spaced over the mesh (capped at the vertex count so no vertex is sampled twice).
std::vector<uint32_t> DesiredSolveVertices(const entt::registry &r, entt::entity e, const ModalSolveSettings &settings, uint32_t num_vertices) {
    if (settings.CopySoundVertices && r.all_of<SoundVertices>(e)) return r.get<const SoundVertices>(e).Vertices;
    const uint32_t ex_count = std::clamp(settings.NumVertices, 1u, num_vertices);
    return iota_view{0u, ex_count} | transform([&](uint32_t i) { return i * num_vertices / ex_count; }) | to<std::vector<uint32_t>>();
}

struct SolveInputs {
    std::vector<vec3> Positions; // Mesh positions at the node's world scale (SI meters)
    std::vector<uint32_t> TriangleIndices;
    std::vector<uint32_t> Vertices; // Excitation vertices
    TetGenOptions TetOptions;
    vec3 NodeScale;
    size_t Hash;
};

SolveInputs BuildSolveInputs(const entt::registry &r, entt::entity e, entt::entity mesh_entity) {
    const auto &settings = r.get<const ModalSolveSettings>(e);
    const auto &mesh = GetMesh(r, mesh_entity);
    const uint32_t num_vertices = mesh.VertexCount();
    const vec3 node_scale = r.get<const WorldTransform>(e).S;
    std::vector<vec3> positions(num_vertices);
    for (uint32_t i = 0; i < num_vertices; ++i) positions[i] = mesh.GetPosition(Mesh::VH{i}) * node_scale;
    auto triangle_indices = mesh.CreateTriangleIndices();
    const TetGenOptions tet_options{.PreserveSurface = true, .Quality = settings.QualityTets, .SimplifyRatio = settings.SolveResolution};
    const auto hash = HashTetInputs(positions, triangle_indices, tet_options);
    return {std::move(positions), std::move(triangle_indices), DesiredSolveVertices(r, e, settings, num_vertices), tet_options, node_scale, hash};
}

// True when the baked model no longer matches the current solve inputs.
bool ModalModelStale(const entt::registry &r, entt::entity e, entt::entity mesh_entity, const SolveInputs &inputs) {
    const auto *summary = r.try_get<const ModalEigenSummary>(e);
    if (!summary) return true;
    if (summary->TetInputsHash != inputs.Hash) return true;
    if (inputs.Vertices != r.get<const ModalModes>(e).Vertices) return true;
    const auto *mat = r.try_get<const AcousticMaterial>(mesh_entity);
    return mat && mat->Properties.PoissonRatio != summary->SolvedMaterial.PoissonRatio;
}

// Launch an async solve for the entity from its current components, unless one is already running
// or the baked model matches the inputs.
void LaunchModalSolve(entt::registry &r, entt::entity viewport, entt::entity e) {
    if (!r.valid(e) || !r.all_of<ModalSolveSettings>(e) || IsSolving(r, e)) return;
    const auto *inst = r.try_get<const Instance>(e);
    if (!inst || !TryGetMesh(r, inst->Entity)) return;
    const auto *material = r.try_get<const AcousticMaterial>(inst->Entity);
    if (!material) return;
    auto inputs = BuildSolveInputs(r, e, inst->Entity);
    if (r.all_of<ModalModes>(e) && !ModalModelStale(r, e, inst->Entity, inputs)) return;

    std::optional<float> fundamental;
    if (const auto path = ActiveSamplePath(r, e)) {
        const auto &frames = GetSampleFrames(r, viewport, *path);
        if (!frames.empty()) fundamental = EstimateFundamentalFrequency(ComputeFft(frames));
    }
    auto excite_positions = inputs.Vertices | transform([&](uint32_t v) { return inputs.Positions[v]; }) | to<std::vector<vec3>>();
    // The most recent solve's basis seeds this one when it was over the same tets.
    std::shared_ptr<const Eigen::MatrixXf> warm_basis;
    if (const auto &warm = r.ctx().get<const ModalWarmStart>(); warm.Basis && warm.TetInputsHash == inputs.Hash) warm_basis = warm.Basis;
    auto work = [inputs = std::move(inputs), material_props = material->Properties, excite_positions = std::move(excite_positions), fundamental, warm_basis = std::move(warm_basis)](JobMonitor &monitor) mutable -> ModalGenerationResult {
        const auto tets = GenerateTets(std::move(inputs.Positions), std::move(inputs.TriangleIndices), inputs.TetOptions);
        if (monitor.Cancelled()) return {};
        auto result = modal::mesh2modes(*tets, material_props, excite_positions, inputs.NodeScale, {.FundamentalFreq = fundamental}, {.SeedBasis = warm_basis.get(), .KeepBasis = true}, &monitor);
        result.Modes.Vertices = std::move(inputs.Vertices);
        result.Modes.BakedScale = inputs.NodeScale;
        result.Summary.TetInputsHash = inputs.Hash;
        auto basis = result.Basis.size() > 0 ? std::make_shared<Eigen::MatrixXf>(std::move(result.Basis)) : nullptr;
        auto model_path = result.Modes.Freqs.empty() ? fs::path{} : SaveModalModelFile({std::move(result.Modes), result.MassProps, BuildTetMeshData(*tets, inputs.NodeScale), std::move(result.Summary)});
        return {std::move(model_path), std::move(basis), inputs.Hash};
    };
    // Intentional registry-ctx write outside Apply: transient background-job bookkeeping.
    r.ctx().get<ModalSolveJobs>().Jobs.push_back(std::make_shared<ModalSolveJob>(e, viewport, Job<ModalGenerationResult>{GetName(r, e), std::move(work)}));
}
} // namespace

void RegisterAudioComponentHandlers(entt::registry &r) {
    RegisterSceneClearHandler(r, [](entt::registry &r) {
        // Drop the modal synthesis bank's slots: the scene's entities are gone, and the next scene's
        // reused entity ids must not retune stale slots. A rebuild follows the next solve or load.
        auto &m = r.ctx().get<ModalAudio>();
        std::scoped_lock lock{m.StructureMutex};
        ClearModalObjects(m);
        // The warm-start basis seeds re-solves of a mesh from the cleared scene, so drop it too.
        r.ctx().get<ModalWarmStart>() = {};
        // In-flight solves target entities from the cleared scene. Their results are discarded on arrival.
        for (auto &job : r.ctx().get<ModalSolveJobs>().Jobs) job->Work.Monitor->RequestCancel();
    });

    // The audio thread reads the registry ctx concurrently (ProcessAudio), so the modal solve slots
    // are created once here and only assigned in place, never erased or re-inserted at runtime.
    r.ctx().emplace<ModalWarmStart>();
    r.ctx().emplace<ModalSolveJobs>();

    track<audio_changes::VertexForce>(r).on<::VertexForce>(On::Create | On::Update | On::Destroy);
    track<audio_changes::ModalModes>(r).on<::ModalModes>(On::Create | On::Update | On::Destroy);
    track<audio_changes::ModalParams>(r)
        .on<ModalGain>(On::Update)
        .on<ModalTuning>(On::Update)
        .on<ModalSoundControls>(On::Create | On::Update);
    track<audio_changes::RecordingStart>(r).on<Recording>(On::Create | On::Update);
    r.ctx().emplace<ModalScaleTracker>().Bind(r);
    track<audio_changes::SoundVerticesDerivation>(r)
        .on<VertexSamples>(On::Create | On::Update | On::Destroy)
        .on<::ModalModes>(On::Create | On::Update | On::Destroy)
        .on<SoundVerticesModel>(On::Create | On::Update | On::Destroy);
    track<audio_changes::ContactDynamicsDerivation>(r)
        .on<MassProperties>(On::Create | On::Update | On::Destroy)
        .on<::ModalModes>(On::Create | On::Update | On::Destroy);
    track<audio_changes::AcousticMaterialEdit>(r).on<AcousticMaterial>(On::Create | On::Update);
    track<audio_changes::AudioConfig>(r).on<AudioOutputConfig>(On::Create | On::Update);
    track<audio_changes::AudioMix>(r).on<AudioOutputMix>(On::Create | On::Update);

    RegisterComponentEventHandler(r, [](entt::registry &r) {
        // Land completed modal solves.
        auto &solve_jobs = r.ctx().get<ModalSolveJobs>().Jobs;
        for (auto it = solve_jobs.begin(); it != solve_jobs.end();) {
            auto &job = **it;
            auto result = job.Work.Poll();
            if (!result) {
                ++it;
                continue;
            }
            if (!job.Work.Monitor->Cancelled()) {
                // Intentional registry-ctx write outside Apply: the warm-start slot is a derived memo, not scene input.
                if (result->Basis) r.ctx().get<ModalWarmStart>() = {result->TetInputsHash, std::move(result->Basis)};
                if (result->ModelPath.empty()) std::cerr << "Modal model computation failed.\n";
                else if (r.valid(job.Entity) && r.all_of<ModalSolveSettings>(job.Entity)) action::ApplyNow(r, job.Viewport, action::audio::ApplyModalModel{job.Entity, std::move(result->ModelPath)});
            }
            it = solve_jobs.erase(it);
        }
        // A material edit rescales every modal model derived from that mesh.
        for (auto mesh_e : reactive<audio_changes::AcousticMaterialEdit>(r)) {
            if (!r.valid(mesh_e) || !r.all_of<AcousticMaterial>(mesh_e)) continue;
            const auto &props = r.get<const AcousticMaterial>(mesh_e).Properties;
            for (auto e : r.view<const ModalEigenSummary, const ::ModalModes, const Instance>()) {
                if (r.get<const Instance>(e).Entity == mesh_e) RescaleModalObject(r, e, props);
            }
        }
        // Rebuild SoundVertices from VertexSamples/ModalModes, selected by SoundVerticesModel.
        // Runs before any handler that reads SoundVertices.
        for (auto e : reactive<audio_changes::SoundVerticesDerivation>(r)) {
            if (!r.valid(e)) continue;
            const auto *model = r.try_get<const SoundVerticesModel>(e);
            std::vector<uint32_t> new_vertices;
            if (model) {
                if (*model == SoundVerticesModel::Samples) {
                    if (const auto *vs = r.try_get<const VertexSamples>(e)) {
                        new_vertices = vs->PathByVertex | std::views::keys | to<std::vector>();
                    }
                } else if (const auto *modes = r.try_get<const ::ModalModes>(e)) {
                    new_vertices = modes->Vertices;
                }
            }
            if (new_vertices.empty()) {
                r.remove<SoundVertices>(e);
                continue;
            }
            if (auto *sv = r.try_get<SoundVertices>(e)) {
                if (sv->Vertices != new_vertices) {
                    r.patch<SoundVertices>(e, [&](auto &sv) { sv.Vertices = std::move(new_vertices); });
                }
            } else {
                r.emplace<SoundVertices>(e, std::move(new_vertices));
            }
            // Ensure MeshActiveElement is valid for the new vertex set.
            const auto mesh_entity = r.get<const Instance>(e).Entity;
            const auto &sv = r.get<const SoundVertices>(e);
            if (const auto *active = r.try_get<const MeshActiveElement>(mesh_entity)) {
                if (!sv.FindVertexIndex(active->Handle)) {
                    r.emplace_or_replace<MeshActiveElement>(mesh_entity, sv.Vertices.front());
                }
            }
        }
        // Refresh contact dynamics before the strike loop below reads them.
        for (auto e : reactive<audio_changes::ContactDynamicsDerivation>(r)) {
            if (r.valid(e)) UpdateContactDynamics(r, e);
        }
        // A created or replaced VertexForce is a strike. Contact pulses are one-shot.
        for (auto e : reactive<audio_changes::VertexForce>(r)) {
            if (!r.valid(e) || !r.all_of<SoundVerticesModel>(e)) continue;
            const auto *vf = r.try_get<::VertexForce>(e);
            if (!vf || vf->Force <= 0) continue;
            const auto &excitable = r.get<const SoundVertices>(e);
            if (auto vi = excitable.FindVertexIndex(vf->Vertex)) {
                r.emplace_or_replace<MeshActiveElement>(r.get<const Instance>(e).Entity, vf->Vertex);
                const auto model = r.get<SoundVerticesModel>(e);
                if (model == SoundVerticesModel::Modal && r.all_of<ModalModes>(e)) {
                    TriggerModalStrike(r, e, *vi, vf->Force, vf->ContactSpeed);
                } else if (model == SoundVerticesModel::Samples && r.all_of<VertexSamples>(e)) {
                    r.patch<VertexSamples>(e, [](auto &s) { s.Play(); });
                }
            }
        }
        // A new recording strikes the object at its active vertex, so the take captures the impact from its onset.
        for (auto e : reactive<audio_changes::RecordingStart>(r)) {
            if (!r.valid(e) || !r.all_of<ModalModes, SoundVertices, Recording>(e)) continue;
            if (r.get<const Recording>(e).Frame == 0) TriggerModalStrike(r, e, GetActiveVertexIndex(r, e), 1.f, 1.f);
        }
        // Reconcile the live output device: a config change re-inits (and may change the negotiated rate),
        // a mix change just applies level/on-off.
        bool device_rate_changed = false;
        if (auto *res = r.ctx().find<AudioDeviceResource>()) {
            if (auto &config_tracker = reactive<audio_changes::AudioConfig>(r); !config_tracker.empty()) {
                const uint32_t prev_rate = res->SampleRate;
                for (auto e : config_tracker) {
                    if (r.valid(e) && r.all_of<AudioOutputConfig, AudioOutputMix>(e)) ConfigureAudioDevice(*res, r.get<const AudioOutputConfig>(e), r.get<const AudioOutputMix>(e));
                }
                device_rate_changed = res->SampleRate != prev_rate;
            }
            for (auto e : reactive<audio_changes::AudioMix>(r)) {
                if (r.valid(e) && r.all_of<AudioOutputMix>(e)) ApplyAudioMix(*res, r.get<const AudioOutputMix>(e));
            }
        }

        auto &modal_tracker = reactive<audio_changes::ModalModes>(r);
        if (!modal_tracker.empty() || device_rate_changed) {
            // Every modal object carries tuning, gain, and solve-settings components (and its mesh an
            // acoustic material), so the synth controls and the modal editor always have state to edit.
            // Intentional registry writes outside Apply: derived defaults for a new model.
            for (auto e : modal_tracker) {
                if (!r.valid(e) || !r.all_of<::ModalModes>(e)) continue;
                const auto &modes = r.get<const ::ModalModes>(e);
                if (!r.all_of<ModalTuning>(e)) r.emplace<ModalTuning>(e, modes.Freqs.empty() ? 0.f : modes.Freqs.front(), 1.f);
                if (!r.all_of<ModalGain>(e)) r.emplace<ModalGain>(e);
                if (!r.all_of<ModalSolveSettings>(e)) {
                    r.emplace<ModalSolveSettings>(e, modes.Vertices.empty() ? ModalSolveSettings{} : ModalSolveSettings{.NumVertices = uint32_t(modes.Vertices.size())});
                }
                if (const auto *inst = r.try_get<const Instance>(e); inst && !r.all_of<AcousticMaterial>(inst->Entity)) {
                    r.emplace<AcousticMaterial>(inst->Entity, materials::acoustic::All.front());
                }
            }
            // A same-layout model replacement (e.g. a material rescale) retunes and reshapes its bank
            // slot in place, without excluding the audio thread. Anything else is structural.
            auto &m = r.ctx().get<ModalAudio>();
            bool rebuild = device_rate_changed;
            for (auto e : modal_tracker) {
                if (rebuild) break;
                const auto *modes = r.valid(e) ? r.try_get<const ::ModalModes>(e) : nullptr;
                const bool active = modes && !modes->Freqs.empty() && r.all_of<SoundVertices>(e);
                const auto slot = FindModalObject(m, e);
                if (!active && !slot) continue;
                if (active && slot && SetModalObjectShapes(m, *slot, *modes)) RetuneModalObject(r, m, *slot, e);
                else rebuild = true;
            }
            if (rebuild) RebuildModalBank(r);
        }
        // Retune objects whose gain or tuning changed, and apply the viewport click level.
        {
            auto &m = r.ctx().get<ModalAudio>();
            for (auto e : reactive<audio_changes::ModalParams>(r)) {
                if (!r.valid(e)) continue;
                if (const auto *controls = r.try_get<const ModalSoundControls>(e)) m.ClickGain = controls->ClickGain;
                if (auto slot = FindModalObject(m, e)) RetuneModalObject(r, m, *slot, e);
            }
            // Retune objects whose node was rescaled.
            auto &scale_tracker = r.ctx().get<ModalScaleTracker>();
            for (auto e : scale_tracker.Storage) {
                if (!r.valid(e)) continue;
                if (auto slot = FindModalObject(m, e)) RetuneModalObject(r, m, *slot, e);
            }
            scale_tracker.Storage.clear();
        }
    });
}

void ProcessAudio(entt::registry &r, entt::entity viewport, float *output, uint32_t frame_count) {
    std::fill_n(output, frame_count, 0.f);
    RenderModal(r.ctx().get<ModalAudio>(), output, frame_count);

    for (const auto [entity, model] : r.view<SoundVerticesModel>().each()) {
        if (model == SoundVerticesModel::Samples) {
            auto *samples = r.try_get<VertexSamples>(entity);
            if (!samples || samples->Stopped) continue;
            const auto path = ActiveSamplePath(r, entity);
            if (!path) continue;
            const auto &impact_samples = GetSampleFrames(r, viewport, *path);
            for (uint32_t i = 0; i < frame_count; ++i) {
                output[i] += samples->Frame < impact_samples.size() ? impact_samples[samples->Frame++] : 0.0f;
            }
        } else if (model == SoundVerticesModel::Modal) {
            if (auto *recording = r.try_get<Recording>(entity)) {
                for (uint32_t i = 0; i < frame_count && !recording->Complete(); ++i) {
                    recording->Record(output[i]);
                }
            }
        }
    }
}

using namespace ImGui;

/***** Sound object *****/

namespace {
constexpr ImVec2 ChartSize{-1, 160};

// If `normalize_max` is set, normalize the data to this maximum value.
void WriteWav(const std::vector<float> &frames, const fs::path &file_path, std::optional<float> normalize_max = {}) {
    static const ma_encoder_config WavEncoderConfig = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 1, SampleRate);
    static ma_encoder WavEncoder;
    if (auto status = ma_encoder_init_file(file_path.c_str(), &WavEncoderConfig, &WavEncoder); status != MA_SUCCESS) {
        throw std::runtime_error(std::format("Failed to initialize wav file {}. Status: {}", file_path.string(), uint(status)));
    }
    const float mult = normalize_max ? *normalize_max / *max_element(frames) : 1.0f;
    const auto frames_normed = frames | transform([mult](float f) { return f * mult; }) | to<std::vector>();
    ma_encoder_write_pcm_frames(&WavEncoder, frames_normed.data(), frames_normed.size(), nullptr);
    ma_encoder_uninit(&WavEncoder);
}

void PlotFrames(const std::vector<float> &frames, std::string_view label = "Waveform", std::optional<uint> highlight_frame = {}) {
    if (ImPlot::BeginPlot(label.data(), ChartSize)) {
        ImPlot::SetupAxes("Frame", "Amplitude");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, frames.size(), ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.1, 1.1, ImGuiCond_Always);
        if (highlight_frame) {
            ImPlot::PlotInfLines("##Highlight", &*highlight_frame, 1, {ImPlotProp_LineColor, ImGui::GetStyleColorVec4(ImGuiCol_PlotLinesHovered)});
        }
        ImPlot::PlotLine("", frames.data(), frames.size());
        ImPlot::EndPlot();
    }
}

void PlotMagnitudeSpectrum(const std::vector<float> &frames, std::string_view label = "Magnitude spectrum", std::optional<float> highlight_freq = {}) {
    static const std::vector<float> *frames_ptr{&frames};
    static FFTData fft{ComputeFft(frames)};
    if (&frames != frames_ptr) {
        fft = ComputeFft(frames);
        frames_ptr = &frames;
    }
    if (ImPlot::BeginPlot(label.data(), ChartSize)) {
        static constexpr float MinDb = -200;
        const uint32_t N = fft.NumReal, N2 = N / 2;
        const float fs = SampleRate; // todo flexible sample rate
        const float fs_n = SampleRate / float(N);
        static std::vector<float> frequency(N2), magnitude(N2);
        frequency.resize(N2);
        magnitude.resize(N2);

        const auto *data = fft.Complex;
        for (uint32_t i = 0; i < N2; i++) {
            frequency[i] = fs_n * float(i);
            magnitude[i] = 20.0f * log10f(sqrtf(data[i][0] * data[i][0] + data[i][1] * data[i][1]) / float(N2));
        }

        ImPlot::SetupAxes("Frequency (Hz)", "Magnitude (dB)");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, fs / 2, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, MinDb, 0, ImGuiCond_Always);
        if (highlight_freq) {
            ImPlot::PlotInfLines("##Highlight", &(*highlight_freq), 1, {ImPlotProp_LineColor, ImGui::GetStyleColorVec4(ImGuiCol_PlotLinesHovered)});
        }
        ImPlot::PlotShaded("", frequency.data(), magnitude.data(), N2, MinDb, {ImPlotProp_FillColor, ImGui::GetStyleColorVec4(ImGuiCol_PlotHistogramHovered)});
        ImPlot::EndPlot();
    }
}

// Returns the index of the hovered mode, if any.
std::optional<size_t> PlotModeData(
    const std::vector<float> &data, std::string_view label, std::string_view x_label, std::string_view y_label,
    std::optional<size_t> highlight_index = {}, std::optional<float> max_value_opt = {}
) {
    std::optional<size_t> hovered_index;
    if (ImPlot::BeginPlot(label.data(), ChartSize)) {
        static constexpr double BarSize = 0.9;
        const float max_value = max_value_opt.value_or(*std::max_element(data.begin(), data.end()));
        ImPlot::SetupAxes(x_label.data(), y_label.data());
        ImPlot::SetupAxesLimits(-0.5f, data.size() - 0.5f, 0, max_value, ImPlotCond_Always);
        if (ImPlot::IsPlotHovered()) {
            if (auto i = std::lround(ImPlot::GetPlotMousePos().x); i >= 0 && i < std::ssize(data)) hovered_index = i;
        }
        if (!highlight_index) {
            ImPlot::PlotBars("", data.data(), data.size(), BarSize);
        } else {
            for (size_t i = 0; i < data.size(); ++i) {
                // Use the first colormap color for the highlighted mode.
                ImPlot::PlotBars(i == *highlight_index ? "##0" : "", &data[i], 1, BarSize, i);
            }
        }
        ImPlot::EndPlot();
    }

    return hovered_index;
}
fs::path PickAudioFile() {
    static const std::array filters{nfdfilteritem_t{"Audio", "wav,mp3,flac,ogg,opus"}};
    nfdchar_t *path = nullptr;
    if (NFD_OpenDialog(&path, filters.data(), filters.size(), "") != NFD_OKAY) return {};
    fs::path file_path{path};
    NFD_FreePath(path);
    return file_path;
}

void DrawModalModelSection(entt::registry &r, entt::entity viewport, entt::entity e, entt::entity mesh_entity) {
    const auto *settings = r.try_get<const ModalSolveSettings>(e);
    const auto *material = r.try_get<const AcousticMaterial>(mesh_entity);
    if (!settings || !material) {
        if (Button("Create modal model")) action::Emit(action::audio::SetupModalModel{});
        return;
    }

    SeparatorText("Material properties");
    if (BeginCombo("Presets", material->Name.c_str())) {
        for (const auto &material_choice : materials::acoustic::All) {
            const bool is_selected = material_choice.Name == material->Name;
            if (Selectable(material_choice.Name.c_str(), is_selected) && !is_selected)
                action::Emit(action::Replace<AcousticMaterial>{.Entity = mesh_entity, .Value = material_choice});
            if (is_selected) SetItemDefaultFocus();
        }
        EndCombo();
    }
    using Props = AcousticMaterialProperties;
    ui::Edit fm{r, mesh_entity};
    fm.Slider<&AcousticMaterial::Properties, &Props::Density>("Density (kg/m^3)", "%.0f");
    fm.Slider<&AcousticMaterial::Properties, &Props::YoungModulus>("Young's modulus (Pa)", "%.3g", ImGuiSliderFlags_Logarithmic);
    fm.Slider<&AcousticMaterial::Properties, &Props::PoissonRatio>("Poisson's ratio", "%.2f");
    fm.Slider<&AcousticMaterial::Properties, &Props::Alpha>("Rayleigh alpha", "%.2f", ImGuiSliderFlags_Logarithmic);
    {
        // Beta's range in seconds sits below the logarithmic slider's zero epsilon, so edit it in microseconds.
        using BetaLimits = FieldLimits<&AcousticMaterial::Properties, &Props::Beta>;
        static constexpr double MinUs{BetaLimits::Min * 1e6}, MaxUs{BetaLimits::Max * 1e6};
        double beta_us = material->Properties.Beta * 1e6;
        const bool changed = SliderScalar("Rayleigh beta (us)", ImGuiDataType_Double, &beta_us, &MinUs, &MaxUs, "%.3f", ImGuiSliderFlags_Logarithmic);
        ui::Gesture(changed, [&] { return action::UpdateOf<&AcousticMaterial::Properties, &Props::Beta>(mesh_entity, beta_us * 1e-6); });
    }

    SeparatorText("Tet mesh");
    ui::Edit fs{r, e};
    fs.Check<&ModalSolveSettings::QualityTets>("Quality");
    MeshEditor::HelpMarker("Add new Steiner points to the interior of the tet mesh to improve model quality.");
    fs.Slider<&ModalSolveSettings::SolveResolution>("Solve resolution");
    MeshEditor::HelpMarker("Fraction of surface triangles used for the modal solve. Lower is faster and less accurate.");

    SeparatorText("Excitation vertices");
    const uint32_t num_vertices = GetMesh(r, mesh_entity).VertexCount();
    const bool has_excitable = r.all_of<SoundVertices>(e);
    if (has_excitable) fs.Check<&ModalSolveSettings::CopySoundVertices>("Copy excitable vertices");
    if (!has_excitable || !settings->CopySoundVertices) {
        const uint32_t min_vertices = 1, max_vertices = num_vertices;
        if (uint32_t v = std::clamp(settings->NumVertices, 1u, num_vertices);
            SliderScalar("Num excitable vertices", ImGuiDataType_U32, &v, &min_vertices, &max_vertices))
            fs.Set<&ModalSolveSettings::NumVertices>(v);
    }

    if (!r.all_of<ModalModes>(e)) {
        if (IsSolving(r, e)) TextDisabled("Solving...");
        else if (Button("Create modal model")) LaunchModalSolve(r, viewport, e);
    }
}

void DrawModalUpdateButton(entt::registry &r, entt::entity viewport, entt::entity e, entt::entity mesh_entity) {
    if (!r.all_of<ModalSolveSettings>(e) || !r.all_of<ModalModes>(e)) return;
    if (IsSolving(r, e)) {
        SameLine();
        TextDisabled("Solving...");
        return;
    }
    if (!ModalModelStale(r, e, mesh_entity, BuildSolveInputs(r, e, mesh_entity))) return;

    SameLine();
    if (Button("Update modal model")) LaunchModalSolve(r, viewport, e);
}

// Mesh vertices targeted by a sample op (Add/Replace/Remove): the active vertex in Excite mode, or the
// selected vertices (edges/faces converted to vertices) in Edit mode. `selection_bits` is ignored outside Edit.
std::vector<uint32_t> GetSampleOpVertices(const entt::registry &r, entt::entity viewport, entt::entity sound_entity, const uint32_t *selection_bits) {
    if (!r.valid(sound_entity)) return {};
    const auto *inst = r.try_get<const Instance>(sound_entity);
    if (!inst) return {};
    const auto mesh_entity = inst->Entity;
    const auto mesh = TryGetMesh(r, mesh_entity);
    if (!mesh) return {};

    const auto mode = r.get<const Interaction>(viewport).Mode;
    if (mode == InteractionMode::Excite) {
        if (const auto *active = r.try_get<const MeshActiveElement>(mesh_entity)) return {active->Handle};
        return {};
    }
    if (mode != InteractionMode::Edit || selection_bits == nullptr) return {};

    const auto *br = r.try_get<const MeshSelectionBitsetRange>(mesh_entity);
    if (!br || br->Count == 0) return {};
    const auto edit_elem = r.get<const EditMode>(viewport).Value;
    auto handles = selection::ScanBitsetRange(selection_bits, br->Offset, br->Count);
    if (edit_elem == Element::Vertex) return handles;
    return selection::ConvertSelectionElement(handles, *mesh, edit_elem, Element::Vertex);
}

// Circular pad returning a position in the unit disk (center = zero). Drag to set, right-click recenters.
bool ImpulseJoystick(vec2 &pos) {
    constexpr float radius{32.f};
    const auto p0 = GetCursorScreenPos();
    InvisibleButton("impulse", {radius * 2, radius * 2}, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    const ImVec2 center{p0.x + radius, p0.y + radius};
    bool changed = false;
    if (IsItemActive() && IsMouseDown(ImGuiMouseButton_Left)) {
        const auto m = GetIO().MousePos;
        pos = {(m.x - center.x) / radius, -(m.y - center.y) / radius};
        if (const float len = glm::length(pos); len > 1.f) pos /= len;
        changed = true;
    } else if (IsItemClicked(ImGuiMouseButton_Right)) {
        pos = {0, 0};
        changed = true;
    }
    auto &dl = *GetWindowDrawList();
    dl.AddCircleFilled(center, radius, GetColorU32(ImGuiCol_FrameBg));
    dl.AddCircle(center, radius, GetColorU32(ImGuiCol_Border));
    dl.AddCircleFilled({center.x + pos.x * radius, center.y - pos.y * radius}, 4.f, GetColorU32(IsItemActive() ? ImGuiCol_SliderGrabActive : ImGuiCol_SliderGrab));
    return changed;
}
} // namespace

void DrawObjectAudioControls(
    entt::registry &r, entt::entity viewport, entt::entity e, entt::entity mesh_entity,
    const uint32_t *selection_bits
) {
    if (e == entt::null || mesh_entity == entt::null) return;

    // Sample ops (Add/Replace/Remove) are only available in Edit / Excite mode.
    const auto mode = r.get<const Interaction>(viewport).Mode;
    const bool sample_ops_available = mode == InteractionMode::Edit || mode == InteractionMode::Excite;
    const auto op_vertices = sample_ops_available ? GetSampleOpVertices(r, viewport, e, selection_bits) : std::vector<uint32_t>{};

    const bool has_model = r.all_of<SoundVerticesModel>(e);
    if (!has_model) {
        SeparatorText("Modal model");
        DrawModalModelSection(r, viewport, e, mesh_entity);
    }

    const auto *samples = r.try_get<const VertexSamples>(e);
    const auto *modal_modes = r.try_get<const ModalModes>(e);
    const auto *excitable = r.try_get<const SoundVertices>(e);
    auto model = has_model ? r.get<SoundVerticesModel>(e) : SoundVerticesModel::Samples;
    const auto *recording = r.try_get<const Recording>(e);
    const uint32_t active_vi = excitable ? GetActiveVertexIndex(r, e) : 0;

    if (samples && modal_modes) {
        PushID("SelectAudioModel");
        auto edit_model = int(model);
        bool model_changed = RadioButton("Recordings", &edit_model, int(SoundVerticesModel::Samples));
        SameLine();
        model_changed |= RadioButton("Modal", &edit_model, int(SoundVerticesModel::Modal));
        PopID();
        if (model_changed) {
            model = SoundVerticesModel(edit_model);
            action::Emit(action::audio::SetModel{model});
        }
    }

    // Cross-model excite section
    if (has_model && excitable) {
        const auto active_vertex = excitable->Vertices[active_vi];
        if (BeginCombo("Vertex", std::to_string(active_vertex).c_str())) {
            for (uint32_t vi = 0; vi < excitable->Vertices.size(); ++vi) {
                if (const auto vertex = excitable->Vertices[vi]; Selectable(std::to_string(vertex).c_str(), vi == active_vi))
                    action::Emit(action::audio::SetExciteVertex{vi, vertex});
            }
            EndCombo();
        }
        const bool can_excite =
            (model == SoundVerticesModel::Samples) ||
            (model == SoundVerticesModel::Modal && (!recording || recording->Complete()));
        if (!can_excite) BeginDisabled();
        Button("Excite");
        if (IsItemActivated()) action::Emit(action::audio::StartExcite{active_vertex});
        else if (IsItemDeactivated()) action::Emit(action::audio::StopExcite{});
        if (!can_excite) EndDisabled();

        if (model == SoundVerticesModel::Modal) {
            TextUnformatted("Impact angle");
            SameLine();
            MeshEditor::HelpMarker("Strike direction relative to the surface.\nCenter hits perpendicular to the surface. Edge hits tangent to the surface.\nRight-click to recenter.");
            ImpulseJoystick(ImpulseAngle);
        }
    }

    // Sample ops + waveform (rendered when in Samples mode or when no model exists yet).
    if (model == SoundVerticesModel::Samples) {
        if (has_model) SeparatorText("Sound samples");
        if (sample_ops_available) {
            std::vector<uint32_t> op_with_sample;
            if (samples) {
                for (const uint32_t mv : op_vertices) {
                    if (samples->PathByVertex.contains(mv)) op_with_sample.push_back(mv);
                }
            }
            const auto n = op_vertices.size(), with_sample = op_with_sample.size();
            if (n == 0) BeginDisabled();
            if (const auto assign_label = n > 1 ? std::format("Assign sample to {} vertices…", n) : std::string{with_sample ? "Replace sample…" : "Assign sample…"};
                Button(assign_label.c_str())) {
                if (auto path = PickAudioFile(); !path.empty()) action::Emit(action::audio::AssignVertexSamples{std::move(op_vertices), std::move(path)});
            }
            if (n == 0) EndDisabled();
            if (with_sample > 0) {
                SameLine();
                if (const auto remove_label = with_sample > 1 ? std::format("Remove {} samples", with_sample) : std::string{"Remove sample"};
                    Button(remove_label.c_str())) {
                    action::Emit(action::audio::RemoveVertexSamples{std::move(op_with_sample)});
                    return;
                }
            }
        }
        if (const auto path = ActiveSamplePath(r, e)) {
            const auto &frames = GetSampleFrames(r, viewport, *path);
            if (!frames.empty()) {
                PlotFrames(frames, "Waveform", samples->Stopped ? std::optional<uint>{} : std::optional{samples->Frame});
                PlotMagnitudeSpectrum(frames, "Spectrum");
            }
        }
    }

    if (!has_model) return;

    SeparatorText("Modal model");
    DrawModalModelSection(r, viewport, e, mesh_entity);

    Spacing();
    if (Button("Delete sound object")) {
        action::Emit(action::audio::DeleteSoundObject{});
        return;
    }
    DrawModalUpdateButton(r, viewport, e, mesh_entity);

    if (model != SoundVerticesModel::Modal) return;

    if (!excitable || !modal_modes) return;

    static std::optional<size_t> hovered_mode_index;
    const auto &modes = *modal_modes;
    if (recording && recording->Complete()) {
        const auto &frames = recording->Frames;
        PlotFrames(frames, "Modal impact waveform");
        const auto highlight_freq = hovered_mode_index ? std::optional{modes.Freqs[*hovered_mode_index]} : std::nullopt;
        PlotMagnitudeSpectrum(frames, "Modal impact spectrum", highlight_freq);
    }

    if (CollapsingHeader("Modal data charts")) {
        std::optional<size_t> new_hovered_index;
        if (auto hovered = PlotModeData(modes.Freqs, "Mode frequencies", "", "Frequency (Hz)", hovered_mode_index)) new_hovered_index = hovered;
        if (auto hovered = PlotModeData(modes.T60s, "Mode T60s", "", "T60 decay time (s)", hovered_mode_index)) new_hovered_index = hovered;
        const auto active_gains = [&]() -> std::vector<float> {
            if (active_vi >= modes.Shapes.size()) return {};
            const auto j = TiltAlongNormal(VertexNormal(GetMesh(r, mesh_entity), excitable->Vertices[active_vi]), ImpulseAngle);
            return modes.Shapes[active_vi] | transform([&](const vec3 &s) { return std::abs(glm::dot(s, j)); }) | to<std::vector<float>>();
        }();
        if (!active_gains.empty()) {
            if (auto hovered = PlotModeData(active_gains, "Mode gains", "Mode index", "Gain", hovered_mode_index)) new_hovered_index = hovered;
        }
        if (hovered_mode_index = new_hovered_index; hovered_mode_index && *hovered_mode_index < modes.Freqs.size()) {
            const auto index = *hovered_mode_index;
            Text(
                "Mode %lu: Freq (scaled) %.2f Hz, Freq (FEM) %.2f, T60 %.2f s, Gain %.4f", index,
                modes.Freqs[index],
                modes.Freqs[index] * modes.OriginalFundamentalFreq / modes.Freqs[0],
                modes.T60s[index],
                index < active_gains.size() ? active_gains[index] : 0.f
            );
        }
    }

    if (CollapsingHeader("Synthesis")) {
        ui::Edit fe{r, e};
        fe.Slider<&ModalGain::Value>("Gain");
        fe.Drag<&ModalTuning::FundamentalFreq>("Fundamental (Hz)", 1.f, "%.1f");
        fe.Slider<&ModalTuning::T60Scale>("T60 scale");
        ui::Edit fv{r, viewport};
        fv.Slider<&ModalSoundControls::ClickGain>("Click");
        MeshEditor::HelpMarker("Level of the rigid-body acceleration-noise click, shared by all objects.");
    }

    const bool is_recording = recording && !recording->Complete();
    if (is_recording) BeginDisabled();
    static constexpr uint32_t RecordFrames = 208'592; // Same length as RealImpact recordings.
    if (Button("Record strike")) action::Emit(action::audio::StartRecording{RecordFrames});
    if (is_recording) EndDisabled();

    if (samples && recording && recording->Complete()) {
        // const auto &modal_fft = ..., &impact_fft = ...;
        // uint32_t ModeCount() const { return modes.Freqs.size(); }
        // const uint32_t n_test_modes = std::min(ModeCount(), 10u);
        // Uncomment to cache `n_test_modes` peak frequencies for display in the spectrum plot.
        // RMSE is abyssmal in most cases...
        // const float rmse = RMSE(GetPeakFrequencies(modal_fft, n_test_modes), GetPeakFrequencies(impact_fft, n_test_modes));
        // Text("RMSE of top %d mode frequencies: %f", n_test_modes, rmse);
        SameLine();
        if (Button("Save wav files")) {
            const auto name = GetName(r, e);
            // Save wav files for both the modal and real-world impact sounds.
            static const auto WavOutDir = fs::path{".."} / "audio_samples";
            WriteWav(recording->Frames, WavOutDir / std::format("{}-modal", name));
            if (const auto path = ActiveSamplePath(r, e)) {
                WriteWav(GetSampleFrames(r, viewport, *path), WavOutDir / std::format("{}-impact", name));
            }
        }
    }
}

void RemoveAudioComponents(entt::registry &r, entt::entity e) {
    CancelModalSolves(r, e);
    r.remove<ScaleLocked, SoundVertices, Recording, SoundVerticesModel, ModalModes, ModalGain, ModalTuning, MassProperties, ContactDynamics, ModalEigenSummary, VertexSamples, ModalSolveSettings, RealImpactActiveMicrophone, RealImpactVertices>(e);
}

void ApplyModalModel(entt::registry &r, entt::entity e, const fs::path &relative_path) {
    if (!r.valid(e) || !r.all_of<Instance>(e)) {
        std::cerr << std::format("Modal model target entity is gone, skipping {}.\n", relative_path.string());
        return;
    }
    auto data = LoadModalModelFile(relative_path);
    if (!data) {
        std::cerr << std::format("Failed to load modal model file {}.\n", (ModalModelsDir() / relative_path).string());
        return;
    }
    const auto mesh_entity = r.get<const Instance>(e).Entity;
    // The material may have been edited while the solve ran. The rescale is exact, so the fresh
    // model lands consistent with the mesh's current material.
    if (const auto *mat = r.try_get<const AcousticMaterial>(mesh_entity); mat && mat->Properties != data->Summary.SolvedMaterial) {
        if (auto rescaled = RescaledModes(data->Summary, data->Modes, mat->Properties)) data->Modes = std::move(*rescaled);
    }
    r.emplace_or_replace<MassProperties>(e, data->Mass);
    ReplaceModalModes(r, e, std::move(data->Modes));
    r.emplace_or_replace<ModalEigenSummary>(e, std::move(data->Summary));
    r.emplace_or_replace<TetMeshData>(mesh_entity, std::move(data->Tets));
    SetModel(r, e, SoundVerticesModel::Modal);
}

void DrawStrikerControls(entt::registry &r, entt::entity viewport) {
    const auto &striker = r.get<const Striker>(viewport);
    // A material change replaces the whole striker (it holds a string); the sliders below edit their fields in place.
    if (BeginCombo("Material", striker.Material.Name.c_str())) {
        for (const auto &choice : materials::acoustic::All) {
            const bool is_selected = choice.Name == striker.Material.Name;
            if (Selectable(choice.Name.c_str(), is_selected) && !is_selected)
                action::Emit(action::Replace<Striker>{.Entity = viewport, .Value = {choice, striker.TipRadius, striker.Length}});
            if (is_selected) SetItemDefaultFocus();
        }
        EndCombo();
    }
    ui::Edit f{r, viewport};
    f.Slider<&Striker::TipRadius>("Tip radius (m)", "%.4f");
    f.Slider<&Striker::Length>("Length (m)", "%.3f");
    Text("Mass: %.3g kg", StrikerMass(striker));
    MeshEditor::HelpMarker("The virtual mallet that strikes objects, a rounded-tip cylinder whose mass comes from its material density and size. A harder material or a shorter (lighter) capsule makes a brighter contact. The tip radius sets the contact curvature.");
}

void DrawModalJobsOverlay(entt::registry &r) {
    const auto &jobs = r.ctx().get<const ModalSolveJobs>().Jobs;
    if (jobs.empty()) return;

    constexpr float Pad{12.f};
    const auto anchor = GetWindowPos() + ImVec2{Pad, GetWindowSize().y - Pad};
    SetNextWindowPos(anchor, ImGuiCond_Always, {0.f, 1.f});
    SetNextWindowBgAlpha(0.85f);
    // The viewport window zeroes its window padding, so restore normal padding for the overlay.
    // Vertical padding matches the item spacing, so each row sits evenly between the window edges and its bar.
    PushStyleVar(ImGuiStyleVar_WindowPadding, {10.f, 6.f});
    PushStyleVar(ImGuiStyleVar_WindowRounding, 6.f);
    PushStyleVar(ImGuiStyleVar_ItemSpacing, {GetStyle().ItemSpacing.x, 6.f});
    PushStyleVar(ImGuiStyleVar_FramePadding, {8.f, 3.f});
    constexpr ImGuiWindowFlags OverlayFlags =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoMove;
    if (Begin("Modal solve jobs", nullptr, OverlayFlags)) {
        // The viewport window covers the overlay whenever it takes focus, so pin the overlay to the display front.
        BringWindowToDisplayFront(GetCurrentWindow());
        for (const auto &job : jobs) {
            auto &monitor = *job->Work.Monitor;
            PushID(job.get());
            BeginGroup();
            AlignTextToFramePadding();
            ImSpinner::SpinnerRotateSegmentsPulsar("##spinner", GetTextLineHeight() * 0.5f, 2.f, GetColorU32(ImGuiCol_Text), 1.1f, 3, 3);
            SameLine();
            TextUnformatted(job->Work.Title.c_str());
            SameLine(0.f, GetStyle().ItemSpacing.x * 3.f);
            const bool cancelled = monitor.Cancelled();
            if (cancelled) BeginDisabled();
            if (Button("Cancel")) monitor.RequestCancel();
            if (cancelled) EndDisabled();
            EndGroup();
            const float progress = monitor.Progress.load(std::memory_order_relaxed);
            ProgressBar(progress > 0.f ? progress : -float(GetTime()), {GetItemRectSize().x, 3.f}, "");
            PopID();
        }
    }
    End();
    PopStyleVar(4);
}
