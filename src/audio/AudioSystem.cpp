#include "AudioSystem.h"
#include "Entity.h"
#include "FFTData.h"
#include "FaustDSP.h"
#include "Instance.h"
#include "Reactive.h"
#include "RealImpactComponents.h"
#include "SceneSelection.h"
#include "SoundVertices.h"
#include "Tets.h"
#include "Transform.h"
#include "Widgets.h" // imgui
#include "Worker.h"
#include "mesh/Mesh.h"
#include "scene_impl/SceneModeComponents.h"

#include "implot.h"
#include "mesh2modes.h"
#include "miniaudio.h"
#include "tetgen.h" // Must be after any Faust includes, since it defined a `REAL` macro.

#include <nfd.h>

#include <iostream>
#include <optional>
#include <print>
#include <ranges>
#include <unordered_map>

using std::ranges::iota_view, std::ranges::to;
using std::views::transform;

static constexpr std::string_view ExciteIndexParamName{"Excite index"}, GateParamName{"Gate"};
static constexpr uint SampleRate = 48'000; // todo respect device sample rate

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

namespace {
const std::vector<float> &GetSampleFrames(const entt::registry &r, entt::entity scene_entity, const fs::path &path) {
    static const std::vector<float> EmptyFrames{};
    if (path.empty()) return EmptyFrames;
    const auto *store = r.try_get<const AudioSamples>(scene_entity);
    if (!store) return EmptyFrames;
    const auto it = store->ByPath.find(path);
    return it != store->ByPath.end() ? it->second.Frames : EmptyFrames;
}

// Inserts frames if `path` is new, otherwise reuses existing frames. Bumps refcount either way.
void AcquireSample(entt::registry &r, entt::entity scene_entity, const fs::path &path, std::vector<float> &&frames) {
    if (path.empty()) return;
    auto &store = r.get_or_emplace<AudioSamples>(scene_entity);
    auto [it, inserted] = store.ByPath.try_emplace(path);
    if (inserted) it->second.Frames = std::move(frames);
    ++it->second.RefCount;
}

// Decrements refcount; erases the entry (and the component if empty) when it hits 0.
void ReleaseSample(entt::registry &r, entt::entity scene_entity, const fs::path &path) {
    if (path.empty()) return;
    auto *store = r.try_get<AudioSamples>(scene_entity);
    if (!store) return;
    const auto it = store->ByPath.find(path);
    if (it == store->ByPath.end()) return;
    if (--it->second.RefCount == 0) store->ByPath.erase(it);
    if (store->ByPath.empty()) r.remove<AudioSamples>(scene_entity);
}
} // namespace

std::vector<float> LoadAudioFrames(const std::string &file_path) {
    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 1, SampleRate);
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
    entt::registry &r, entt::entity scene_entity, entt::entity e,
    std::span<const uint32_t> mesh_vertices, fs::path path, std::vector<float> &&frames
) {
    if (mesh_vertices.empty() || path.empty()) return;
    auto &vs = r.get_or_emplace<VertexSamples>(e);
    vs.Stop();

    // `frames` is consumed on the first new-path insertion; subsequent AcquireSample calls
    // see the path already in the store and only bump refcount, so the moved-from vector is ignored.
    bool vs_changed = false;
    for (uint32_t mv : mesh_vertices) {
        auto [it, inserted] = vs.PathByVertex.try_emplace(mv, path);
        if (!inserted) {
            if (it->second == path) continue;
            ReleaseSample(r, scene_entity, it->second);
            it->second = path;
        }
        AcquireSample(r, scene_entity, path, std::move(frames));
        vs_changed = true;
    }
    if (vs_changed) r.patch<VertexSamples>(e, [](auto &) {});
    if (!r.all_of<SoundVerticesModel>(e)) r.emplace<SoundVerticesModel>(e, SoundVerticesModel::Samples);
}

void RemoveVertexSamples(
    entt::registry &r, entt::entity scene_entity, entt::entity e,
    std::span<const uint32_t> mesh_vertices
) {
    auto *vs = r.try_get<VertexSamples>(e);
    if (!vs || mesh_vertices.empty()) return;
    vs->Stop();
    bool vs_changed = false;
    for (uint32_t mv : mesh_vertices) {
        const auto it = vs->PathByVertex.find(mv);
        if (it == vs->PathByVertex.end()) continue;
        ReleaseSample(r, scene_entity, it->second);
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
    entt::registry &r, entt::entity scene_entity, entt::entity e,
    std::span<const uint32_t> mesh_vertices, std::vector<LoadedSample> &&samples
) {
    for (size_t i = 0; i < samples.size() && i < mesh_vertices.size(); ++i) {
        AssignVertexSample(r, scene_entity, e, {&mesh_vertices[i], 1}, std::move(samples[i].first), std::move(samples[i].second));
    }
}

// Recording of an excitation.
struct Recording {
    Recording(uint frame_count) : Frames(frame_count) {}

    std::vector<float> Frames;
    uint Frame{0};

    bool Complete() const { return Frame == Frames.size(); }
    void Record(float value) {
        if (!Complete()) Frames[Frame++] = value;
    }
};

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

/***** Faust DSP code generation *****/

// Per-entity component: generated Faust DSP code for a modal model.
struct ModalDsp {
    std::string Name, Definition, Eval;
};

namespace {
// Generate DSP code from modal modes.
std::string GenerateModalModelDsp(const ModalModes &modes, std::string_view model_name, bool freq_control) {
    const auto &freqs = modes.Freqs;
    const auto &gains = modes.Gains;
    const auto &t60s = modes.T60s;
    const auto n_modes = freqs.size();
    const auto n_ex_pos = gains.size();

    std::stringstream dsp;
    dsp << model_name << "(" << (freq_control ? "freq," : "")
        << "exPos,t60Scale) = _ <: "
           "par(mode,nModes,pm.modeFilter(modeFreqs(mode),modesT60s(mode),"
           "modesGains(int(exPos),mode))) :> /(nModes)\n"
        << "with{\n"
        << "nModes = " << n_modes << ";\n";
    if (n_ex_pos > 1) dsp << "nExPos = " << n_ex_pos << ";\n";

    dsp << "modeFreqsUnscaled(n) = waveform{";
    for (size_t mode = 0; mode < n_modes; ++mode) {
        dsp << freqs[mode];
        if (mode < n_modes - 1) dsp << ",";
    }
    dsp << "},int(n) : rdtable;\n";
    dsp << "modeFreqs(mode) = ";
    if (freq_control) dsp << "freq*modeFreqsUnscaled(mode)/modeFreqsUnscaled(0)";
    else dsp << "modeFreqsUnscaled(mode)";
    dsp << ";\n";

    dsp << "modesGains(p,n) = waveform{";
    for (size_t ex_pos = 0; ex_pos < gains.size(); ++ex_pos) {
        for (size_t mode = 0; mode < gains[ex_pos].size(); ++mode) {
            dsp << gains[ex_pos][mode];
            if (mode < n_modes - 1) dsp << ",";
        }
        if (ex_pos < gains.size() - 1) dsp << ",";
    }
    dsp << "},int(p*nModes+n) : rdtable" << (freq_control ? " : select2(modeFreqs(n)<(ma.SR/2-1),0)" : "") << ";\n";

    dsp << "modesT60s(n) = t60Scale * (waveform{";
    for (size_t mode = 0; mode < n_modes; ++mode) {
        dsp << t60s[mode];
        if (mode < n_modes - 1) dsp << ",";
    }
    dsp << "},int(n) : rdtable);\n";
    dsp << "};\n";

    return dsp.str();
}

// Note: Frequency control is not physically accurate as it doesn't scale mode dampings.
ModalDsp GenerateModalDsp(std::string_view model_name, const ModalModes &modes, const SoundVertices &sound_vertices, bool freq_control) {
    const float fundamental_freq = modes.Freqs.front();
    static constexpr std::string_view ModelGain{"gain = hslider(\"Gain[scale:log]\",0.2,0,0.5,0.01)"};
    static constexpr std::string_view ModelT60Scale{"t60Scale = hslider(\"t60[scale:log][tooltip: Scale T60 decay values of all modes by the same amount.]\",1,0.1,10,0.01)"};
    const auto model_freq = std::format("freq = hslider(\"Frequency[scale:log][tooltip: Fundamental frequency of the model]\",{},50,16000,1)", fundamental_freq);
    const uint num_excitable = sound_vertices.Vertices.size();
    const auto model_ex_pos = std::format("exPos = nentry(\"{}\",{},0,{},1)", ExciteIndexParamName, (num_excitable - 1) / 2, num_excitable - 1);
    const auto model = GenerateModalModelDsp(modes, "modalModel", freq_control);
    const auto model_definition = std::format("{} = environment {{\n{}\n{};\n{};\n{};\n{};\n}};", model_name, model, ModelGain, model_freq, model_ex_pos, ModelT60Scale);

    constexpr std::string_view ToSAH{" : ba.sAndH(gate)"}; // add a sample and hold on the gate in serial
    const auto freq = freq_control ? std::format("{}.freq{},", model_name, ToSAH) : "";
    const auto model_eval = std::format("{}.modalModel({}{}.exPos{},{}.t60Scale{})*{}.gain", model_name, freq, model_name, ToSAH, model_name, ToSAH, model_name);
    return {std::string(model_name), model_definition, model_eval};
}

std::string GenerateDsp(entt::registry &r) {
    auto view = r.view<const ModalDsp>();
    if (view.empty()) return "";

    static const auto HammerGate = std::format("gate = button(\"../../{}[tooltip: Applies the current parameters and excites the vertex.]\")", GateParamName);
    static constexpr std::string_view HammerHardness{"hammerHardness = hslider(\"Hammer hardness\",0.9,0,1,0.01)"};
    static constexpr std::string_view HammerSize{"hammerSize = hslider(\"Hammer size\",0.1,0,1,0.01)"};
    static constexpr std::string_view Hammer{"hammer(trig,hardness,size) = en.ar(att,att,trig)*no.noise : fi.lowpass(3,ctoff)\nwith{ ctoff = (1-size)*9500+500; att = (1-hardness)*0.01+0.001; }"};
    static constexpr std::string_view HammerEval = "hammer(gate,hammerHardness,hammerSize)";
    static const auto HammerDefinition = std::format("{};\n{};\n{};\n{};", HammerGate, HammerHardness, HammerSize, Hammer);

    const auto count = view.size();
    std::stringstream modal_definitions;
    for (auto [_, dsp] : view.each()) modal_definitions << dsp.Definition << "\n\n";

    const bool multiple_models = count > 1;
    std::stringstream models_eval;
    if (multiple_models) models_eval << "tgroup(\"Models\", ";
    size_t i = 0;
    for (auto [_, dsp] : view.each()) {
        if (i > 0) models_eval << ",";
        models_eval << std::format("vgroup(\"{}\", {})", dsp.Name, dsp.Eval);
        ++i;
    }
    if (multiple_models) models_eval << ")";

    const auto switch_definition = multiple_models ? std::format("N={};\nswitchN(n, s) = par(i, n , _*(i==s));\nmodelIndex = nentry(\"Excite model\", 0, 0, N-1, 1);", count) : "";
    const std::string_view switch_eval = multiple_models ? "switchN(N, modelIndex) : " : "";
    const std::string_view mix = multiple_models ? "/(N)" : "_";
    auto hammer = std::format("vgroup(\"Hammer\", {})", HammerEval);
    // Since models are in a tab group when there are multiple, the gate param needs to be one level deeper for its path to match.
    if (multiple_models) hammer = std::format("vgroup(\"\", {})", hammer);
    return std::format(
        "import(\"stdfaust.lib\");\n\n{}\n\n{}\n\n{}\n\nprocess = {} <: {}{} :> {};\n",
        HammerDefinition, switch_definition, modal_definitions.str(), hammer, switch_eval, models_eval.str(), mix
    );
}

/***** Free functions for sound object control *****/

void Stop(entt::registry &r, entt::entity scene_entity, entt::entity e) {
    if (auto *samples = r.try_get<VertexSamples>(e)) samples->Stop();
    if (r.all_of<ModalModes>(e)) r.get<FaustDSP>(scene_entity).Set(GateParamName, 0);
}

void SetModel(entt::registry &r, entt::entity scene_entity, entt::entity e, SoundVerticesModel model) {
    Stop(r, scene_entity, e);

    const bool is_sample = model == SoundVerticesModel::Samples && r.all_of<VertexSamples>(e);
    const bool is_modal = model == SoundVerticesModel::Modal && r.all_of<ModalModes>(e);
    if (!is_sample && !is_modal) return;

    r.emplace_or_replace<SoundVerticesModel>(e, model);
}

void SetVertex(entt::registry &r, entt::entity scene_entity, entt::entity e, uint vertex) {
    Stop(r, scene_entity, e);
    if (r.all_of<ModalModes>(e)) r.get<FaustDSP>(scene_entity).Set(ExciteIndexParamName, vertex);
}

void SetVertexForce(entt::registry &r, entt::entity scene_entity, entt::entity e, float force) {
    const auto model = r.get<SoundVerticesModel>(e);
    // Update vertex force in the active model.
    if (model == SoundVerticesModel::Samples && force > 0) {
        if (r.all_of<VertexSamples>(e)) r.patch<VertexSamples>(e, [](auto &s) { s.Play(); });
    } else if (model == SoundVerticesModel::Modal && r.all_of<ModalModes>(e)) {
        r.get<FaustDSP>(scene_entity).Set(GateParamName, force);
    }
}

// Reactive change types for audio system
namespace audio_changes {
struct VertexForce {};
struct ModalModes {};
struct SoundVerticesDerivation {};
} // namespace audio_changes
} // namespace

void RegisterAudioComponentHandlers(entt::registry &r, entt::entity scene_entity) {
    track<audio_changes::VertexForce>(r).on<::VertexForce>(On::Create | On::Destroy);
    track<audio_changes::ModalModes>(r).on<::ModalModes>(On::Create | On::Destroy);
    track<audio_changes::SoundVerticesDerivation>(r)
        .on<VertexSamples>(On::Create | On::Update | On::Destroy)
        .on<::ModalModes>(On::Create | On::Update | On::Destroy)
        .on<SoundVerticesModel>(On::Create | On::Update | On::Destroy);

    RegisterComponentEventHandler(r, [scene_entity](entt::registry &r) {
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
        for (auto e : reactive<audio_changes::VertexForce>(r)) {
            if (!r.valid(e) || !r.all_of<SoundVerticesModel>(e)) continue;
            if (const auto *vf = r.try_get<::VertexForce>(e)) {
                const auto &excitable = r.get<const SoundVertices>(e);
                if (auto vi = excitable.FindVertexIndex(vf->Vertex)) {
                    r.emplace_or_replace<MeshActiveElement>(r.get<const Instance>(e).Entity, vf->Vertex);
                    SetVertex(r, scene_entity, e, *vi);
                    SetVertexForce(r, scene_entity, e, vf->Force);
                }
            } else {
                SetVertexForce(r, scene_entity, e, 0.f);
            }
        }
        auto &modal_tracker = reactive<audio_changes::ModalModes>(r);
        if (!modal_tracker.empty()) {
            for (auto e : modal_tracker) {
                if (r.valid(e) && r.all_of<::ModalModes>(e)) {
                    auto name = GetName(r, e);
                    std::ranges::replace(name, ' ', '_');
                    r.emplace_or_replace<ModalDsp>(e, GenerateModalDsp(name, r.get<const ::ModalModes>(e), r.get<const SoundVertices>(e), true));
                } else if (r.valid(e)) {
                    r.remove<ModalDsp>(e);
                }
            }
            r.get<FaustDSP>(scene_entity).SetCode(GenerateDsp(r));
        }
    });
}

void ProcessAudio(FaustDSP &dsp, entt::registry &r, entt::entity scene_entity, AudioBuffer buffer) {
    dsp.Compute(buffer.FrameCount, &buffer.Input, &buffer.Output);

    for (const auto [entity, model] : r.view<SoundVerticesModel>().each()) {
        if (model == SoundVerticesModel::Samples) {
            auto *samples = r.try_get<VertexSamples>(entity);
            if (!samples || samples->Stopped) continue;
            const auto path = ActiveSamplePath(r, entity);
            if (!path) continue;
            const auto &impact_samples = GetSampleFrames(r, scene_entity, *path);
            for (uint i = 0; i < buffer.FrameCount; ++i) {
                buffer.Output[i] += samples->Frame < impact_samples.size() ? impact_samples[samples->Frame++] : 0.0f;
            }
        } else if (model == SoundVerticesModel::Modal) {
            if (auto *recording = r.try_get<Recording>(entity)) {
                if (recording->Frame == 0) dsp.Set(GateParamName, 1);
                if (!recording->Complete()) {
                    for (uint i = 0; i < buffer.FrameCount && !recording->Complete(); ++i) {
                        recording->Record(buffer.Output[i]);
                    }
                    if (recording->Complete()) dsp.Set(GateParamName, 0);
                }
            }
        }
    }
}

using namespace ImGui;

/***** Sound object *****/

namespace {
constexpr void ApplyCosineWindow(float *w, uint n, const float *coeff, uint ncoeff) {
    if (n == 1) {
        w[0] = 1.0;
        return;
    }

    const uint wlength = n;
    for (uint i = 0; i < n; ++i) {
        float wi = 0.0;
        for (uint j = 0; j < ncoeff; ++j) wi += coeff[j] * __cospi(float(2 * i * j) / float(wlength));
        w[i] = wi;
    }
}

// Create Blackman-Harris window
constexpr std::vector<float> CreateBlackmanHarris(uint n) {
    std::vector<float> window(n);
    static constexpr float coeff[4] = {0.35875, -0.48829, 0.14128, -0.01168};
    ApplyCosineWindow(window.data(), n, coeff, sizeof(coeff) / sizeof(float));
    return window;
}

constexpr std::vector<float> ApplyWindow(const std::vector<float> &window, const float *data) {
    std::vector<float> windowed(window.size());
    for (uint i = 0; i < window.size(); ++i) windowed[i] = window[i] * data[i];
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
    std::ranges::nth_element(upper, upper.begin() + upper.size() / 2);
    const float threshold = upper[upper.size() / 2] + 15.f;

    constexpr size_t W = 15; // Prominence window
    const size_t min_bin = static_cast<size_t>(50.0f * fft.NumReal / SampleRate);
    for (size_t i = std::max(min_bin, W); i < n_bins - W; ++i) {
        // Local maximum?
        if (mag_db[i] <= mag_db[i - 1] || mag_db[i] <= mag_db[i + 1]) continue;
        if (mag_db[i] < threshold) continue;

        constexpr float ProminenceThresholdDb{10.f};
        // Prominence check: peak must be above the local mean by ProminenceThresholdDb
        float local_sum = 0;
        for (size_t j = i - W; j <= i + W; ++j) local_sum += mag_db[j];
        const float local_mean = local_sum / (2 * W + 1);
        if (mag_db[i] - local_mean < ProminenceThresholdDb) continue;
        return float(i) * SampleRate / fft.NumReal;
    }
    return std::nullopt;
}

constexpr float LinearToDb(float linear) { return 20.0f * log10f(linear); }
constexpr ImVec2 ChartSize{-1, 160};

// Capture a short audio segment shortly after the impact for FFT.
FFTData ComputeFft(const std::vector<float> &frames) {
    static constexpr uint FftStartFrame = 30, FftEndFrame = SampleRate / 16;
    static const auto BHWindow = CreateBlackmanHarris(FftEndFrame - FftStartFrame);
    return {ApplyWindow(BHWindow, frames.data() + FftStartFrame)};
}

// If `normalize_max` is set, normalize the data to this maximum value.
void WriteWav(const std::vector<float> &frames, fs::path file_path, std::optional<float> normalize_max = {}) {
    static ma_encoder_config WavEncoderConfig = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 1, SampleRate);
    static ma_encoder WavEncoder;
    if (auto status = ma_encoder_init_file(file_path.c_str(), &WavEncoderConfig, &WavEncoder); status != MA_SUCCESS) {
        throw std::runtime_error(std::format("Failed to initialize wav file {}. Status: {}", file_path.string(), uint(status)));
    }
    const float mult = normalize_max ? *normalize_max / *std::ranges::max_element(frames) : 1.0f;
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
        const uint N = fft.NumReal, N2 = N / 2;
        const float fs = SampleRate; // todo flexible sample rate
        const float fs_n = SampleRate / float(N);
        static std::vector<float> frequency(N2), magnitude(N2);
        frequency.resize(N2);
        magnitude.resize(N2);

        const auto *data = fft.Complex;
        for (uint i = 0; i < N2; i++) {
            frequency[i] = fs_n * float(i);
            magnitude[i] = LinearToDb(sqrtf(data[i][0] * data[i][0] + data[i][1] * data[i][1]) / float(N2));
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
} // namespace

namespace {
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
            if (auto i = size_t(ImPlot::GetPlotMousePos().x + 0.5f); i >= 0 && i < data.size()) hovered_index = i;
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
std::unique_ptr<Worker<ModalModes>> DspGenerator;

LoadedSample PickAndLoadAudio() {
    static const std::array filters{nfdfilteritem_t{"Audio", "wav,mp3,flac,ogg,opus"}};
    nfdchar_t *path = nullptr;
    if (NFD_OpenDialog(&path, filters.data(), filters.size(), "") != NFD_OKAY) return {};
    fs::path file_path{path};
    auto frames = LoadAudioFrames(path);
    NFD_FreePath(path);
    return {std::move(file_path), std::move(frames)};
}

// Render the modal model create/edit form. Assumes ModalModelCreateInfo is present on `e`.
// `modal_modes` may be null (e.g. bare-mesh first-time create).
void DrawModalCreateForm(
    entt::registry &r, entt::entity SceneEntity, entt::entity e, entt::entity mesh_entity,
    ImGuiWindow *parent_window, bool has_excitable, const ModalModes *modal_modes
) {
    using namespace ImGui;
    auto &info = r.get<ModalModelCreateInfo>(e);
    if (!BeginChild("CreateModalAudioModel", ImVec2{-FLT_MIN, 0.f}, ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY, ImGuiWindowFlags_MenuBar)) {
        EndChild();
        return;
    }
    if (BeginMenuBar()) {
        Text("Create modal audio model");
        EndMenuBar();
    }
    SeparatorText("Material properties");
    if (BeginCombo("Presets", info.Material.Name.c_str())) {
        for (const auto &material_choice : materials::acoustic::All) {
            const bool is_selected = (material_choice.Name == info.Material.Name);
            if (Selectable(material_choice.Name.c_str(), is_selected)) info.Material = material_choice;
            if (is_selected) SetItemDefaultFocus();
        }
        EndCombo();
    }
    Text("Density (kg/m^3)");
    InputDouble("##Density", &info.Material.Properties.Density, 0.0f, 0.0f, "%.3f");
    Text("Young's modulus (Pa)");
    InputDouble("##Young's modulus", &info.Material.Properties.YoungModulus, 0.0f, 0.0f, "%.3f");
    Text("Poisson's ratio");
    InputDouble("##Poisson's ratio", &info.Material.Properties.PoissonRatio, 0.0f, 0.0f, "%.3f");
    Text("Rayleigh damping alpha/beta");
    InputDouble("##Rayleigh damping alpha", &info.Material.Properties.Alpha, 0.0f, 0.0f, "%.3f");
    InputDouble("##Rayleigh damping beta", &info.Material.Properties.Beta, 0.0f, 0.0f, "%.3f");

    SeparatorText("Tet mesh");
    Checkbox("Quality", &info.QualityTets);
    MeshEditor::HelpMarker("Add new Steiner points to the interior of the tet mesh to improve model quality.");

    SeparatorText("SoundVertices vertices");
    if (has_excitable) Checkbox("Copy excitable vertices", &info.CopySoundVertices);
    if (!has_excitable || !info.CopySoundVertices) {
        const uint num_vertices = r.get<Mesh>(mesh_entity).VertexCount();
        info.NumVertices = std::min(info.NumVertices, num_vertices);
        const uint min_vertices = 1, max_vertices = num_vertices;
        SliderScalar("Num excitable vertices", ImGuiDataType_U32, &info.NumVertices, &min_vertices, &max_vertices);
    }

    if (Button(modal_modes ? "Update" : "Create")) {
        Stop(r, SceneEntity, e);
        r.emplace_or_replace<AcousticMaterial>(mesh_entity, info.Material);
        const auto &mesh = r.get<const Mesh>(mesh_entity);
        const auto num_vertices = mesh.VertexCount();
        auto new_sound_vertices = info.CopySoundVertices && r.all_of<SoundVertices>(e) ?
            r.get<const SoundVertices>(e) :
            SoundVertices{iota_view{0u, uint(info.NumVertices)} | transform([&](uint i) { return i * num_vertices / info.NumVertices; }) | to<std::vector<uint>>()};
        constexpr float ScaleFactor{2}; // Mode freq estimates for RealImpact meshes seem to be consistently about twice as high as recordings.
        auto tets = GenerateTets(mesh, ScaleFactor * r.get<const Transform>(e).S, {.PreserveSurface = true, .Quality = info.QualityTets});
        std::optional<float> fundamental;
        if (const auto path = ActiveSamplePath(r, e)) {
            const auto &frames = GetSampleFrames(r, SceneEntity, *path);
            if (!frames.empty()) fundamental = EstimateFundamentalFrequency(ComputeFft(frames));
        }
        auto material_props = info.Material.Properties;
        DspGenerator = std::make_unique<Worker<ModalModes>>(
            parent_window, "Generating modal audio model...",
            [tets = std::move(tets), material_props, sound_vertices = std::move(new_sound_vertices), fundamental]() mutable {
                return m2f::mesh2modes(*tets, material_props, sound_vertices.Vertices, fundamental);
            }
        );
        r.remove<ModalModelCreateInfo>(e);
    }
    SameLine();
    if (Button("Cancel")) r.remove<ModalModelCreateInfo>(e);
    EndChild();
}
} // namespace

void DrawObjectAudioControls(
    entt::registry &r, entt::entity SceneEntity, entt::entity e, entt::entity mesh_entity,
    const uint32_t *selection_bits
) {
    if (e == entt::null || mesh_entity == entt::null) return;

    if (auto &dsp_generator = DspGenerator) {
        if (auto modes = dsp_generator->Render()) {
            dsp_generator.reset();
            if (modes->Freqs.empty()) {
                std::cerr << "Modal model computation failed.\n";
            } else {
                if (!r.all_of<ScaleLocked>(e)) r.emplace<ScaleLocked>(e);
                uint32_t excite_idx = 0;
                if (const auto *active = r.try_get<const MeshActiveElement>(mesh_entity)) {
                    const auto &verts = modes->Vertices;
                    if (auto it = std::ranges::find(verts, active->Handle); it != verts.end()) {
                        excite_idx = std::ranges::distance(verts.begin(), it);
                    }
                }
                r.emplace_or_replace<ModalModes>(e, std::move(*modes));
                r.get<FaustDSP>(SceneEntity).Set(ExciteIndexParamName, excite_idx);
                SetModel(r, SceneEntity, e, SoundVerticesModel::Modal);
            }
        }
    }

    using namespace ImGui;

    // Modal create/edit form takes over whenever open, regardless of whether the entity is a sound object yet.
    if (r.all_of<ModalModelCreateInfo>(e)) {
        const auto *modal_modes = r.try_get<const ModalModes>(e);
        DrawModalCreateForm(r, SceneEntity, e, mesh_entity, GetCurrentWindow(), r.all_of<SoundVertices>(e), modal_modes);
        return;
    }

    // Sample ops (Add/Replace/Remove) are only available in Edit / Excite mode.
    const auto mode = r.get<const SceneInteraction>(SceneEntity).Mode;
    const bool sample_ops_available = mode == InteractionMode::Edit || mode == InteractionMode::Excite;
    const auto op_vertices = sample_ops_available ? scene_selection::GetSampleOpVertices(r, SceneEntity, e, selection_bits) : std::vector<uint32_t>{};

    const bool has_model = r.all_of<SoundVerticesModel>(e);
    if (!has_model && Button("Create modal model")) {
        ModalModelCreateInfo info{};
        if (const auto *material = r.try_get<const AcousticMaterial>(mesh_entity)) info.Material = *material;
        r.emplace<ModalModelCreateInfo>(e, std::move(info));
        return;
    }

    auto *samples = r.try_get<VertexSamples>(e);
    const auto *modal_modes = r.try_get<ModalModes>(e);
    const auto *excitable = r.try_get<const SoundVertices>(e);
    auto model = has_model ? r.get<SoundVerticesModel>(e) : SoundVerticesModel::Samples;
    auto *recording = r.try_get<Recording>(e);
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
            SetModel(r, SceneEntity, e, model);
        }
    }

    // Cross-model excite section
    if (has_model && excitable) {
        const auto active_vertex = excitable->Vertices[active_vi];
        if (BeginCombo("Vertex", std::to_string(active_vertex).c_str())) {
            for (uint vi = 0; vi < excitable->Vertices.size(); ++vi) {
                const auto vertex = excitable->Vertices[vi];
                if (Selectable(std::to_string(vertex).c_str(), vi == active_vi)) {
                    r.remove<VertexForce>(e);
                    r.emplace_or_replace<MeshActiveElement>(mesh_entity, vertex);
                    SetVertex(r, SceneEntity, e, vi);
                }
            }
            EndCombo();
        }
        const bool can_excite =
            (model == SoundVerticesModel::Samples) ||
            (model == SoundVerticesModel::Modal && (!recording || recording->Complete()));
        if (!can_excite) BeginDisabled();
        Button("Excite");
        if (IsItemActivated()) {
            r.remove<VertexForce>(e);
            r.emplace<VertexForce>(e, active_vertex, 1.f);
        } else if (IsItemDeactivated()) r.remove<VertexForce>(e);
        if (!can_excite) EndDisabled();
    }

    // Sample ops + waveform (rendered when in Samples mode or when no model exists yet).
    if (model == SoundVerticesModel::Samples) {
        if (has_model) SeparatorText("Sound samples");
        if (sample_ops_available) {
            std::vector<uint32_t> op_with_sample;
            if (samples) {
                for (uint32_t mv : op_vertices) {
                    if (samples->PathByVertex.contains(mv)) op_with_sample.push_back(mv);
                }
            }
            const auto n = op_vertices.size(), with_sample = op_with_sample.size();
            if (n == 0) BeginDisabled();
            const auto assign_label = n > 1 ? std::format("Assign sample to {} vertices…", n) : std::string{with_sample ? "Replace sample…" : "Assign sample…"};
            if (Button(assign_label.c_str())) {
                auto [path, frames] = PickAndLoadAudio();
                if (!frames.empty()) AssignVertexSample(r, SceneEntity, e, op_vertices, std::move(path), std::move(frames));
            }
            if (n == 0) EndDisabled();
            if (with_sample > 0) {
                SameLine();
                const auto remove_label = with_sample > 1 ? std::format("Remove {} samples", with_sample) : std::string{"Remove sample"};
                if (Button(remove_label.c_str())) {
                    RemoveVertexSamples(r, SceneEntity, e, op_with_sample);
                    return;
                }
            }
        }
        if (const auto path = ActiveSamplePath(r, e)) {
            const auto &frames = GetSampleFrames(r, SceneEntity, *path);
            if (!frames.empty()) {
                PlotFrames(frames, "Waveform", samples->Stopped ? std::optional<uint>{} : std::optional{samples->Frame});
                PlotMagnitudeSpectrum(frames, "Spectrum");
            }
        }
    }

    if (!has_model) return;

    SeparatorText("Modal model");
    if (Button(std::format("{} modal model", modal_modes ? "Edit" : "Create").c_str())) {
        ModalModelCreateInfo info{};
        if (modal_modes && excitable) info.NumVertices = excitable->Vertices.size();
        if (const auto *material = r.try_get<const AcousticMaterial>(mesh_entity)) info.Material = *material;
        r.emplace<ModalModelCreateInfo>(e, std::move(info));
    }

    Spacing();
    if (Button("Delete sound object")) {
        RemoveAudioComponents(r, e);
        return;
    }

    if (model != SoundVerticesModel::Modal) return;

    // Modal
    if (!excitable || !modal_modes) return;

    static std::optional<size_t> hovered_mode_index;
    const auto &modes = *modal_modes;
    if (recording && recording->Complete()) {
        const auto &frames = recording->Frames;
        PlotFrames(frames, "Modal impact waveform");
        const auto highlight_freq = hovered_mode_index ? std::optional{modes.Freqs[*hovered_mode_index]} : std::nullopt;
        PlotMagnitudeSpectrum(frames, "Modal impact spectrum", highlight_freq);
    }

    // Poll the Faust DSP UI to see if the current excitation vertex has changed.
    const auto excite_index = uint(r.get<FaustDSP>(SceneEntity).Get(ExciteIndexParamName));
    if (active_vi != excite_index && excite_index < excitable->Vertices.size()) {
        r.emplace_or_replace<MeshActiveElement>(mesh_entity, excitable->Vertices[excite_index]);
    }
    if (CollapsingHeader("Modal data charts")) {
        std::optional<size_t> new_hovered_index;
        if (auto hovered = PlotModeData(modes.Freqs, "Mode frequencies", "", "Frequency (Hz)", hovered_mode_index)) new_hovered_index = hovered;
        if (auto hovered = PlotModeData(modes.T60s, "Mode T60s", "", "T60 decay time (s)", hovered_mode_index)) new_hovered_index = hovered;
        if (auto hovered = PlotModeData(modes.Gains[active_vi], "Mode gains", "Mode index", "Gain", hovered_mode_index, 1.f)) new_hovered_index = hovered;
        if (hovered_mode_index = new_hovered_index; hovered_mode_index && *hovered_mode_index < modes.Freqs.size()) {
            const auto index = *hovered_mode_index;
            Text(
                "Mode %lu: Freq (scaled) %.2f Hz, Freq (FEM) %.2f, T60 %.2f s, Gain %.2f dB", index,
                modes.Freqs[index],
                modes.Freqs[index] * modes.OriginalFundamentalFreq / modes.Freqs[0],
                modes.T60s[index],
                modes.Gains[active_vi][index]
            );
        }
    }

    if (CollapsingHeader("DSP parameters")) r.get<FaustDSP>(SceneEntity).DrawParams();
    if (CollapsingHeader("DSP graph")) r.get<FaustDSP>(SceneEntity).DrawGraph();
    if (Button("Print DSP code")) std::println("DSP code:\n\n{}\n", r.get<FaustDSP>(SceneEntity).GetCode());

    const bool is_recording = recording && !recording->Complete();
    if (is_recording) BeginDisabled();
    static constexpr uint RecordFrames = 208'592; // Same length as RealImpact recordings.
    if (Button("Record strike")) recording = &r.emplace<Recording>(e, RecordFrames);
    if (is_recording) EndDisabled();

    if (samples && recording && recording->Complete()) {
        // const auto &modal_fft = ..., &impact_fft = ...;
        // uint ModeCount() const { return modes.Freqs.size(); }
        // const uint n_test_modes = std::min(ModeCount(), 10u);
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
                WriteWav(GetSampleFrames(r, SceneEntity, *path), WavOutDir / std::format("{}-impact", name));
            }
        }
    }
}

void RemoveAudioComponents(entt::registry &r, entt::entity e) {
    r.remove<ScaleLocked, SoundVertices, Recording, SoundVerticesModel, ModalModes, ModalDsp, VertexSamples, ModalModelCreateInfo, RealImpactActiveMicrophone, RealImpactVertices>(e);
}
