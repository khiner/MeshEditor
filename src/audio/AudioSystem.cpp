#include "AudioSystem.h"
#include "Entity.h"
#include "FFTData.h"
#include "FaustDSP.h"
#include "Instance.h"
#include "Reactive.h"
#include "RealImpactComponents.h"
#include "SoundVertices.h"
#include "Tets.h"
#include "Transform.h"
#include "Widgets.h" // imgui
#include "Worker.h"
#include "mesh/Mesh.h"

#include "implot.h"
#include "mesh2modes.h"
#include "miniaudio.h"
#include "tetgen.h" // Must be after any Faust includes, since it defined a `REAL` macro.

#include <nfd.h>

#include <iostream>
#include <optional>
#include <print>
#include <ranges>

using std::ranges::find, std::ranges::find_if, std::ranges::iota_view, std::ranges::sort, std::ranges::to;
using std::views::transform, std::views::take;

static constexpr std::string_view ExciteIndexParamName{"Excite index"}, GateParamName{"Gate"};
static constexpr uint SampleRate = 48'000; // todo respect device sample rate

struct VertexSamples {
    std::vector<std::vector<float>> Frames; // Frames[vertex_index][frame_index]
    std::vector<uint32_t> Vertices; // Mesh vertex indices corresponding to each frame index in Frames
    uint32_t Frame{0};
    bool Stopped{true}; // Don't immediately play

    const std::vector<float> &GetFrames(uint32_t selected_vertex_index) const { return Frames[selected_vertex_index]; }
    bool Complete(uint32_t selected_vertex_index) const { return Stopped || Frame >= GetFrames(selected_vertex_index).size(); }
    void Stop() { Stopped = true; }
    void Play() {
        Frame = 0;
        Stopped = false;
    }
};

void SetVertexSamples(entt::registry &r, entt::entity e, std::vector<std::vector<float>> &&frames) {
    const auto &vertices = r.get<const SoundVertices>(e).Vertices;
    if (auto *samples = r.try_get<VertexSamples>(e)) {
        samples->Stop();
        samples->Frames = std::move(frames);
        samples->Vertices = vertices;
    } else if (!frames.empty()) {
        r.emplace<VertexSamples>(e, std::move(frames), std::vector<uint32_t>(vertices));
        r.emplace_or_replace<SoundVerticesModel>(e, SoundVerticesModel::Samples);
    }
}

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

void AddVertexSample(entt::registry &r, entt::entity e, uint32_t mesh_vertex, std::vector<float> &&frames) {
    if (!r.all_of<SoundVertices>(e)) r.emplace<SoundVertices>(e);
    if (!r.all_of<VertexSamples>(e)) r.emplace<VertexSamples>(e);
    auto &vs = r.get<VertexSamples>(e);
    auto &sv = r.get<SoundVertices>(e);
    vs.Stop();
    if (auto it = std::ranges::find(vs.Vertices, mesh_vertex); it != vs.Vertices.end()) {
        vs.Frames[std::ranges::distance(vs.Vertices.begin(), it)] = std::move(frames);
    } else {
        vs.Vertices.push_back(mesh_vertex);
        vs.Frames.emplace_back(std::move(frames));
    }
    if (std::ranges::find(sv.Vertices, mesh_vertex) == sv.Vertices.end()) {
        sv.Vertices.push_back(mesh_vertex);
        r.patch<SoundVertices>(e, [](auto &) {});
    }
    if (!r.all_of<SoundVerticesModel>(e)) r.emplace<SoundVerticesModel>(e, SoundVerticesModel::Samples);
}

void RemoveVertexSample(entt::registry &r, entt::entity e, uint32_t mesh_vertex) {
    auto *vs = r.try_get<VertexSamples>(e);
    if (!vs) return;
    const auto it = std::ranges::find(vs->Vertices, mesh_vertex);
    if (it == vs->Vertices.end()) return;
    const auto i = std::ranges::distance(vs->Vertices.begin(), it);
    vs->Stop();
    vs->Vertices.erase(it);
    vs->Frames.erase(vs->Frames.begin() + i);
    const bool has_modal = r.all_of<ModalModes>(e);
    if (!has_modal) {
        if (auto *sv = r.try_get<SoundVertices>(e)) {
            if (auto sv_it = std::ranges::find(sv->Vertices, mesh_vertex); sv_it != sv->Vertices.end()) {
                sv->Vertices.erase(sv_it);
                r.patch<SoundVertices>(e, [](auto &) {});
            }
        }
    }
    if (vs->Vertices.empty()) {
        if (has_modal) r.remove<VertexSamples>(e);
        else RemoveAudioComponents(r, e);
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

    const auto *samples = r.try_get<const VertexSamples>(e);
    const auto *modal = r.try_get<const ModalModes>(e);
    const bool is_sample = model == SoundVerticesModel::Samples && samples;
    const bool is_modal = model == SoundVerticesModel::Modal && modal;
    if (!is_sample && !is_modal) return;

    r.emplace_or_replace<SoundVerticesModel>(e, model);
    const auto &vertices = is_sample ? samples->Vertices : modal->Vertices;
    r.patch<SoundVertices>(e, [&vertices](auto &sv) { sv.Vertices = vertices; });
    // Ensure MeshActiveElement is valid for the new vertex set.
    const auto mesh_entity = r.get<const Instance>(e).Entity;
    if (const auto *active = r.try_get<const MeshActiveElement>(mesh_entity)) {
        if (!r.get<const SoundVertices>(e).FindVertexIndex(active->Handle)) {
            r.emplace_or_replace<MeshActiveElement>(mesh_entity, vertices.front());
        }
    }
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
} // namespace audio_changes
} // namespace

void RegisterAudioComponentHandlers(entt::registry &r, entt::entity scene_entity) {
    track<audio_changes::VertexForce>(r).on<::VertexForce>(On::Create | On::Destroy);
    track<audio_changes::ModalModes>(r).on<::ModalModes>(On::Create | On::Destroy);

    RegisterComponentEventHandler(r, [scene_entity](entt::registry &r) {
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
                    const auto &modes = r.get<const ::ModalModes>(e);
                    const auto &sv = r.get<const SoundVertices>(e);
                    auto name = GetName(r, e);
                    std::ranges::replace(name, ' ', '_');
                    r.emplace_or_replace<ModalDsp>(e, GenerateModalDsp(name, modes, sv, true));
                } else if (r.valid(e)) {
                    r.remove<ModalDsp>(e);
                }
            }
            r.get<FaustDSP>(scene_entity).SetCode(GenerateDsp(r));
        }
    });
}

void ProcessAudio(FaustDSP &dsp, entt::registry &r, AudioBuffer buffer) {
    dsp.Compute(buffer.FrameCount, &buffer.Input, &buffer.Output);

    for (const auto [entity, model] : r.view<SoundVerticesModel>().each()) {
        if (model == SoundVerticesModel::Samples) {
            if (auto *samples = r.try_get<VertexSamples>(entity);
                samples && !samples->Frames.empty() && !samples->Stopped) {
                const auto &impact_samples = samples->GetFrames(GetActiveVertexIndex(r, entity));
                for (uint i = 0; i < buffer.FrameCount; ++i) {
                    buffer.Output[i] += samples->Frame < impact_samples.size() ? impact_samples[samples->Frame++] : 0.0f;
                }
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
struct ModalModelResult {
    ModalModes Modes;
    SoundVertices SoundVertices;
};
std::unique_ptr<Worker<ModalModelResult>> DspGenerator;

std::vector<float> PickAndLoadAudio() {
    static const std::array filters{nfdfilteritem_t{"Audio", "wav,mp3,flac,ogg,opus"}};
    nfdchar_t *path = nullptr;
    if (NFD_OpenDialog(&path, filters.data(), filters.size(), "") != NFD_OKAY) return {};
    auto frames = LoadAudioFrames(path);
    NFD_FreePath(path);
    return frames;
}

// Render the modal model create/edit form. Assumes ModalModelCreateInfo is present on `e`.
// `excitable`/`modal_modes`/`samples` may be null (e.g. bare-mesh first-time create).
void DrawModalCreateForm(
    entt::registry &r, entt::entity SceneEntity, entt::entity e, entt::entity mesh_entity,
    ImGuiWindow *parent_window,
    const SoundVertices *excitable, const ModalModes *modal_modes, const VertexSamples *samples, uint32_t active_vi
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
    if (excitable) Checkbox("Copy excitable vertices", &info.CopySoundVertices);
    if (!excitable || !info.CopySoundVertices) {
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
        if (samples && excitable) fundamental = EstimateFundamentalFrequency(ComputeFft(samples->GetFrames(active_vi)));
        auto material_props = info.Material.Properties;
        DspGenerator = std::make_unique<Worker<ModalModelResult>>(
            parent_window, "Generating modal audio model...",
            [tets = std::move(tets), material_props, sound_vertices = std::move(new_sound_vertices), fundamental]() mutable {
                auto modes = m2f::mesh2modes(*tets, material_props, sound_vertices.Vertices, fundamental);
                return ModalModelResult{.Modes = std::move(modes), .SoundVertices = std::move(sound_vertices)};
            }
        );
        r.remove<ModalModelCreateInfo>(e);
    }
    SameLine();
    if (Button("Cancel")) r.remove<ModalModelCreateInfo>(e);
    EndChild();
}
} // namespace

void DrawObjectAudioControls(entt::registry &r, entt::entity SceneEntity, entt::entity e, entt::entity mesh_entity) {
    if (e == entt::null || mesh_entity == entt::null) return;

    if (auto &dsp_generator = DspGenerator) {
        if (auto result = dsp_generator->Render()) {
            dsp_generator.reset();
            if (result->Modes.Freqs.empty()) {
                std::cerr << "Modal model computation failed.\n";
            } else {
                if (!r.all_of<ScaleLocked>(e)) r.emplace<ScaleLocked>(e);
                r.emplace_or_replace<SoundVertices>(e, std::move(result->SoundVertices));
                r.emplace_or_replace<ModalModes>(e, std::move(result->Modes));
                r.get<FaustDSP>(SceneEntity).Set(ExciteIndexParamName, GetActiveVertexIndex(r, e));
                SetModel(r, SceneEntity, e, SoundVerticesModel::Modal);
            }
        }
    }

    using namespace ImGui;

    // Modal create/edit form takes over whenever open, regardless of whether the entity
    // is a sound object yet. ScaleLocked + SoundVerticesModel::Modal are added only when
    // the background DspGenerator actually completes.
    if (r.all_of<ModalModelCreateInfo>(e)) {
        const auto *excitable = r.try_get<const SoundVertices>(e);
        const auto *modal_modes = r.try_get<const ModalModes>(e);
        const auto *samples = r.try_get<const VertexSamples>(e);
        const uint32_t active_vi = excitable ? GetActiveVertexIndex(r, e) : 0;
        DrawModalCreateForm(r, SceneEntity, e, mesh_entity, GetCurrentWindow(), excitable, modal_modes, samples, active_vi);
        return;
    }

    // Entity may not yet be a sound object — offer entry points.
    if (!r.all_of<SoundVerticesModel>(e)) {
        if (Button("Create modal model")) {
            ModalModelCreateInfo info{};
            if (const auto *material = r.try_get<const AcousticMaterial>(mesh_entity)) info.Material = *material;
            r.emplace<ModalModelCreateInfo>(e, std::move(info));
            return;
        }
        if (Button("Add vertex sample…")) {
            auto frames = PickAndLoadAudio();
            if (!frames.empty()) {
                const auto vertex = r.all_of<MeshActiveElement>(mesh_entity) ? r.get<MeshActiveElement>(mesh_entity).Handle : 0u;
                AddVertexSample(r, e, vertex, std::move(frames));
            }
        }
        return;
    }

    const auto *samples = r.try_get<VertexSamples>(e);
    const auto *modal_modes = r.try_get<ModalModes>(e);
    auto model = r.get<SoundVerticesModel>(e);
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
    auto *recording = r.try_get<Recording>(e);
    const auto *excitable = r.try_get<const SoundVertices>(e);
    const uint32_t active_vi = excitable ? GetActiveVertexIndex(r, e) : 0;
    if (excitable) {
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

    if (model == SoundVerticesModel::Samples && excitable) {
        SeparatorText("Sound samples");
        const auto active_vertex = excitable->Vertices[active_vi];
        const bool has_sample = samples && active_vi < samples->Frames.size() && !samples->Frames[active_vi].empty();
        if (Button(has_sample ? "Replace sample…" : "Add sample…")) {
            auto frames = PickAndLoadAudio();
            if (!frames.empty()) AddVertexSample(r, e, active_vertex, std::move(frames));
        }
        if (has_sample) {
            SameLine();
            if (Button("Remove sample")) {
                RemoveVertexSample(r, e, active_vertex);
                return;
            }
        }
        if (has_sample) {
            const auto &frames = samples->GetFrames(active_vi);
            PlotFrames(frames, "Waveform", samples->Stopped ? std::optional<uint>{} : std::optional{samples->Frame});
            PlotMagnitudeSpectrum(frames, "Spectrum");
        }
    }

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
            WriteWav(samples->GetFrames(active_vi), WavOutDir / std::format("{}-impact", name));
        }
    }
}

void RemoveAudioComponents(entt::registry &r, entt::entity e) {
    r.remove<ScaleLocked, SoundVertices, Recording, SoundVerticesModel, ModalModes, ModalDsp, VertexSamples, ModalModelCreateInfo, RealImpactActiveMicrophone>(e);
}
