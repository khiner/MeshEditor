#include "SoundObject.h"

#include "AudioBuffer.h"
#include "Excitable.h"
#include "FFTData.h"
#include "FaustDSP.h"
#include "Registry.h"
#include "Scale.h"
#include "Tets.h"
#include "Widgets.h" // imgui
#include "Worker.h"
#include "mesh/Mesh.h"
#include "mesh2faust.h"

#include "implot.h"
#include "miniaudio.h"
#include "tetMesh.h" // Vega
#include "tetgen.h" // Must be after any Faust includes, since it defined a `REAL` macro.
#include <entt/entity/registry.hpp>

#include <format>
#include <print>
#include <ranges>

using std::ranges::find, std::ranges::iota_view, std::ranges::sort, std::ranges::to;
using std::views::transform, std::views::take;

class tetgenio;

struct Mesh2FaustResult {
    std::string ModelDsp; // Faust DSP code defining the model function.
    std::vector<float> ModeFreqs; // Mode frequencies
    std::vector<float> ModeT60s; // Mode T60 decay times
    std::vector<std::vector<float>> ModeGains; // Mode gains by [exitation position][mode]
    std::vector<uint32_t> ExcitableVertices; // Excitable vertices
    AcousticMaterialProperties Material;
};

namespace {
struct ImpactRecording {
    static constexpr uint FrameCount = 208592; // Same length as RealImpact recordings.
    float Frames[FrameCount];
    uint Frame{0};
    bool Complete{false};
};

constexpr uint SampleRate = 48'000; // todo respect device sample rate

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
    std::vector<float> windowed_data(window.size());
    for (uint i = 0; i < window.size(); ++i) windowed_data[i] = window[i] * data[i];
    return windowed_data;
}

// Ordered by lowest to highest frequency.
constexpr std::vector<float> FindPeakFrequencies(const fftwf_complex *data, uint n_bins, uint n_peaks) {
    const uint N_2 = n_bins / 2;

    std::vector<std::pair<float, uint>> peaks; // (magnitude, bin)
    for (uint i = 1; i < N_2 - 1; i++) {
        float mag_sq = data[i][0] * data[i][0] + data[i][1] * data[i][1];
        float left_mag_sq = data[i - 1][0] * data[i - 1][0] + data[i - 1][1] * data[i - 1][1];
        float right_mag_sq = data[i + 1][0] * data[i + 1][0] + data[i + 1][1] * data[i + 1][1];
        if (mag_sq > left_mag_sq && mag_sq > right_mag_sq) peaks.emplace_back(mag_sq, i);
    }

    // Sort descending by magnitude and convert to frequency.
    sort(peaks, [](const auto &a, const auto &b) { return a.first > b.first; });
    auto peak_freqs = peaks | take(n_peaks) |
        transform([n_bins](const auto &p) { return float(p.second * SampleRate / n_bins); }) | to<std::vector>();
    sort(peak_freqs);
    return peak_freqs;
}

constexpr float LinearToDb(float linear) { return 20.0f * log10f(linear); }
constexpr ImVec2 ChartSize = {-1, 160};

struct Waveform {
    // Capture a short audio segment shortly after the impact for FFT.
    static constexpr uint FftStartFrame = 30, FftEndFrame = SampleRate / 16;
    inline static const auto BHWindow = CreateBlackmanHarris(FftEndFrame - FftStartFrame);
    const std::vector<float> Frames;
    std::vector<float> WindowedFrames;
    const FFTData FftData;
    // std::vector<float> PeakFrequencies;

    Waveform(const float *frames, uint frame_count)
        : Frames(frames, frames + frame_count),
          WindowedFrames(ApplyWindow(BHWindow, Frames.data() + FftStartFrame)), FftData(WindowedFrames) {}

    void PlotFrames(std::string_view label = "Waveform", std::optional<uint> highlight_frame = {}) const {
        if (ImPlot::BeginPlot(label.data(), ChartSize)) {
            ImPlot::SetupAxes("Frame", "Amplitude");
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, Frames.size(), ImGuiCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, -1.1, 1.1, ImGuiCond_Always);

            if (highlight_frame) {
                ImPlot::PushStyleColor(ImPlotCol_Line, ImGui::GetStyleColorVec4(ImGuiCol_PlotLinesHovered));
                ImPlot::PlotInfLines("##Highlight", &*highlight_frame, 1);
                ImPlot::PopStyleColor();
            }
            ImPlot::PushStyleVar(ImPlotStyleVar_Marker, ImPlotMarker_None);
            ImPlot::PlotLine("", Frames.data(), Frames.size());
            ImPlot::PopStyleVar();
            ImPlot::EndPlot();
        }
    }

    void PlotMagnitudeSpectrum(std::string_view label = "Magnitude spectrum", std::optional<uint> highlight_peak_freq_index = {}) const {
        if (ImPlot::BeginPlot(label.data(), ChartSize)) {
            static constexpr float MIN_DB = -200;
            const FFTData &fft = FftData;
            const uint N = WindowedFrames.size();
            const uint N_2 = N / 2;
            const float fs = SampleRate; // todo flexible sample rate
            const float fs_n = SampleRate / float(N);

            static std::vector<float> frequency(N_2), magnitude(N_2);
            frequency.resize(N_2);
            magnitude.resize(N_2);

            const auto *data = fft.Complex;
            for (uint i = 0; i < N_2; i++) {
                frequency[i] = fs_n * float(i);
                magnitude[i] = LinearToDb(sqrtf(data[i][0] * data[i][0] + data[i][1] * data[i][1]) / float(N_2));
            }

            ImPlot::SetupAxes("Frequency (Hz)", "Magnitude (dB)");
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, fs / 2, ImGuiCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, MIN_DB, 0, ImGuiCond_Always);
            ImPlot::PushStyleVar(ImPlotStyleVar_Marker, ImPlotMarker_None);
            ImPlot::PushStyleColor(ImPlotCol_Fill, ImGui::GetStyleColorVec4(ImGuiCol_PlotHistogramHovered));
            (void)highlight_peak_freq_index; // unused
            // Disabling peak frequency display for now.
            // for (uint i = 0; i < PeakFrequencies.size(); ++i) {
            //     const bool is_highlighted = highlight_peak_freq_index && i == *highlight_peak_freq_index;
            //     const float freq = PeakFrequencies[i];
            //     if (is_highlighted) {
            //         ImPlot::PushStyleColor(ImPlotCol_Line, ImGui::GetStyleColorVec4(ImGuiCol_PlotLinesHovered));
            //         ImPlot::PlotInfLines("##Highlight", &freq, 1);
            //         ImPlot::PopStyleColor();
            //     } else {
            //         ImPlot::PlotInfLines("##Peak", &freq, 1);
            //     }
            // }
            ImPlot::PlotShaded("", frequency.data(), magnitude.data(), N_2, MIN_DB);
            ImPlot::PopStyleColor();
            ImPlot::PopStyleVar();
            ImPlot::EndPlot();
        }
    }

    std::vector<float> GetPeakFrequencies(uint n_peaks) { return FindPeakFrequencies(FftData.Complex, WindowedFrames.size(), n_peaks); }

    float GetMaxValue() const { return *std::max_element(Frames.begin(), Frames.end()); }

    // If `normalize_max` is set, normalize the data to this maximum value.
    void WriteWav(fs::path file_path, std::optional<float> normalize_max = std::nullopt) const {
        if (auto status = ma_encoder_init_file(file_path.c_str(), &WavEncoderConfig, &WavEncoder); status != MA_SUCCESS) {
            throw std::runtime_error(std::format("Failed to initialize wav file {}. Status: {}", file_path.string(), uint(status)));
        }
        const float mult = normalize_max ? *normalize_max / *std::max_element(Frames.begin(), Frames.end()) : 1.0f;
        const auto frames = Frames | transform([mult](float f) { return f * mult; }) | to<std::vector>();
        ma_encoder_write_pcm_frames(&WavEncoder, frames.data(), frames.size(), nullptr);
        ma_encoder_uninit(&WavEncoder);
    }

private:
    inline static ma_encoder_config WavEncoderConfig = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 1, SampleRate);
    inline static ma_encoder WavEncoder;
};
} // namespace

// All model-specific data.
struct ImpactAudioModel {
    ImpactAudioModel(std::vector<std::vector<float>> &&impact_frames, std::vector<uint> &&vertex_indices)
        : ImpactFrames(std::move(impact_frames)),
          Excitable(std::move(vertex_indices)),
          // All samples are the same length.
          MaxFrame(ImpactFrames.empty() ? 0 : ImpactFrames.front().size()) {
        UpdateWaveform();
    }
    ~ImpactAudioModel() = default;

    const ImpactAudioModel &operator=(ImpactAudioModel &&other) noexcept {
        if (this != &other) {
            ImpactFrames = std::move(other.ImpactFrames);
            Excitable = std::move(other.Excitable);
            Frame = other.Frame;
            Waveform = std::move(other.Waveform);
        }
        return *this;
    }

    std::vector<std::vector<float>> ImpactFrames;
    Excitable Excitable;
    uint MaxFrame;
    uint Frame{MaxFrame}; // Start at the end, so it doesn't immediately play.
    std::unique_ptr<Waveform> Waveform; // Selected vertex's waveform

    void ProduceAudio(const AudioBuffer &buffer) {
        if (ImpactFrames.empty()) return;

        const auto &impact_samples = ImpactFrames[Excitable.SelectedVertexIndex];
        // todo - resample from 48kHz to device sample rate if necessary
        for (uint i = 0; i < buffer.FrameCount; ++i) {
            buffer.Output[i] += Frame < impact_samples.size() ? impact_samples[Frame++] : 0.0f;
        }
    }

    void Start() { Frame = 0; }
    void Stop() { Frame = MaxFrame; }
    bool IsStarted() const { return Frame != MaxFrame; }

    bool CanExcite() const { return bool(Waveform); }
    void SetVertex(uint vertex) {
        Excitable.SelectVertex(vertex);
        UpdateWaveform();
    }
    void SetVertexIndex(uint vertex_index) {
        Excitable.SelectedVertexIndex = vertex_index;
        UpdateWaveform();
    }
    void SetImpactFrames(std::vector<std::vector<float>> &&impact_frames) {
        if (ImpactFrames.size() != impact_frames.size()) return;

        ImpactFrames = std::move(impact_frames);
        UpdateWaveform();
    }

    void SetVertexForce(float force) {
        if (force > 0 && !IsStarted()) Start();
        else if (force == 0 && IsStarted()) Stop();
    }

    void Draw() const {
        if (!Waveform) return;

        Waveform->PlotFrames("Real-world impact waveform", Frame);
        Waveform->PlotMagnitudeSpectrum("Real-world impact spectrum");
    }

private:
    void UpdateWaveform() {
        Stop();
        if (Excitable.SelectedVertexIndex < ImpactFrames.size()) {
            const auto &frames = ImpactFrames[Excitable.SelectedVertexIndex];
            Waveform = std::make_unique<::Waveform>(frames.data(), frames.size());
        }
    }
};

namespace {
// Returns the index of the hovered mode, if any.
std::optional<size_t> PlotModeData(
    const std::vector<float> &data, std::string_view label, std::string_view x_label, std::string_view y_label,
    std::optional<size_t> highlight_index = std::nullopt, std::optional<float> max_value_opt = std::nullopt
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
std::unique_ptr<Worker<Mesh2FaustResult>> DspGenerator;
} // namespace

struct ModalAudioModel {
    ModalAudioModel(FaustDSP &dsp, Mesh2FaustResult &&m2f)
        : Dsp(dsp), M2F(std::move(m2f)), Excitable(M2F.ExcitableVertices) {
        M2F.ExcitableVertices.clear(); // These are only used to populate `Excitable`.
        SetVertex(Excitable.SelectedVertex());
    }

    ~ModalAudioModel() = default;

    uint ModeCount() const { return M2F.ModeFreqs.size(); }

    void ProduceAudio(AudioBuffer &buffer) const {
        if (!ImpactRecording) return;

        if (ImpactRecording->Frame == 0) SetParam(GateParamName, 1);
        if (!ImpactRecording->Complete) {
            for (uint i = 0; i < buffer.FrameCount && ImpactRecording->Frame < ImpactRecording::FrameCount; ++i, ++ImpactRecording->Frame) {
                ImpactRecording->Frames[ImpactRecording->Frame] = buffer.Output[i];
            }
            if (ImpactRecording->Frame == ImpactRecording::FrameCount) {
                ImpactRecording->Complete = true;
                SetParam(GateParamName, 0);
            }
        }
    }

    bool CanExcite() const { return !ImpactRecording || ImpactRecording->Complete; }
    void SetVertex(uint vertex) {
        Stop();
        Excitable.SelectVertex(vertex);
        const auto &vertices = Excitable.ExcitableVertices;
        if (auto it = find(vertices, vertex); it != vertices.end()) {
            SetParam(ExciteIndexParamName, std::ranges::distance(vertices.begin(), it));
        }
    }
    void SetVertexForce(float force) { SetParam(GateParamName, force); }
    void Stop() { SetVertexForce(0); }

    void SetParam(std::string_view param_label, Sample param_value) const {
        Dsp.Set(std::move(param_label), param_value);
    }

    void Draw() {
        using namespace ImGui;

        const bool is_recording = ImpactRecording && !ImpactRecording->Complete;
        if (is_recording) BeginDisabled();
        SameLine();
        if (Button("Record strike")) ImpactRecording = std::make_unique<::ImpactRecording>();
        if (is_recording) EndDisabled();
        if (ImpactRecording && ImpactRecording->Complete) {
            Waveform = std::make_unique<::Waveform>(ImpactRecording->Frames, ImpactRecording::FrameCount);
            ImpactRecording.reset();
        }
        if (Waveform) {
            Waveform->PlotFrames("Modal impact waveform");
            Waveform->PlotMagnitudeSpectrum("Modal impact spectrum", HoveredModeIndex);
        }

        // Poll the Faust DSP UI to see if the current excitation vertex has changed.
        Excitable.SelectedVertexIndex = uint(Dsp.Get(ExciteIndexParamName));
        if (CollapsingHeader("Modal data charts")) {
            std::optional<size_t> new_hovered_index;
            if (auto hovered = PlotModeData(M2F.ModeFreqs, "Mode frequencies", "", "Frequency (Hz)", HoveredModeIndex)) new_hovered_index = hovered;
            if (auto hovered = PlotModeData(M2F.ModeT60s, "Mode T60s", "", "T60 decay time (s)", HoveredModeIndex)) new_hovered_index = hovered;
            if (auto hovered = PlotModeData(M2F.ModeGains[Excitable.SelectedVertexIndex], "Mode gains", "Mode index", "Gain", HoveredModeIndex, 1.f)) new_hovered_index = hovered;
            if (HoveredModeIndex = new_hovered_index; HoveredModeIndex && *HoveredModeIndex < M2F.ModeFreqs.size()) {
                const auto index = *HoveredModeIndex;
                Text(
                    "Mode %lu: Freq %.2f Hz, T60 %.2f s, Gain %.2f dB", index,
                    M2F.ModeFreqs[index],
                    M2F.ModeT60s[index],
                    M2F.ModeGains[Excitable.SelectedVertexIndex][index]
                );
            }
        }

        if (CollapsingHeader("DSP parameters")) Dsp.DrawParams();
        if (CollapsingHeader("DSP graph")) Dsp.DrawGraph();
        if (Button("Print DSP code")) std::println("DSP code:\n\n{}\n", Dsp.GetCode());
    }

    std::unique_ptr<Waveform> Waveform{}; // Recorded waveform

    FaustDSP &Dsp;
    Mesh2FaustResult M2F;
    Excitable Excitable;

private:
    std::unique_ptr<ImpactRecording> ImpactRecording;
    std::optional<size_t> HoveredModeIndex;
};

/*
namespace {
// Assumes a and b are the same length.
constexpr float RMSE(const std::vector<float> &a, const std::vector<float> &b) {
    float sum = 0;
    for (size_t i = 0; i < a.size(); ++i) sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrtf(sum / a.size());
}
} // namespace
*/

SoundObject::SoundObject(AcousticMaterial material, FaustDSP &dsp) : Dsp(dsp), Material(std::move(material)) {}
SoundObject::~SoundObject() = default;

void SoundObject::SetImpactFrames(std::vector<std::vector<float>> &&impact_frames, std::vector<uint> &&vertex_indices) {
    if (!impact_frames.empty()) {
        ImpactModel = std::make_unique<ImpactAudioModel>(std::move(impact_frames), std::move(vertex_indices));
    }
}
void SoundObject::SetImpactFrames(std::vector<std::vector<float>> &&impact_frames) {
    if (ImpactModel) {
        ImpactModel->SetImpactFrames(std::move(impact_frames));
    }
}

void SoundObject::ProduceAudio(AudioBuffer &buffer) const {
    if (Model == SoundObjectModel::ImpactAudio && ImpactModel) {
        ImpactModel->ProduceAudio(buffer);
    } else if (Model == SoundObjectModel::Modal && ModalModel) {
        ModalModel->ProduceAudio(buffer);
    }
}

void SoundObject::SetVertex(uint vertex) {
    // Update vertex in all present models.
    if (ImpactModel) ImpactModel->SetVertex(vertex);
    if (ModalModel) ModalModel->SetVertex(vertex);
}
void SoundObject::SetVertexForce(float force) {
    // Update vertex force in the active model.
    if (Model == SoundObjectModel::ImpactAudio && ImpactModel) ImpactModel->SetVertexForce(force);
    else if (Model == SoundObjectModel::Modal && ModalModel) ModalModel->SetVertexForce(force);
}

void SoundObject::SetModel(SoundObjectModel model, entt::registry &r, entt::entity entity) {
    if (ImpactModel) ImpactModel->Stop();
    if (ModalModel) ModalModel->Stop();
    Model = model;
    r.emplace_or_replace<Excitable>(entity, GetExcitable());
}

const Excitable &SoundObject::GetExcitable() const {
    if (Model == SoundObjectModel::ImpactAudio && ImpactModel) return ImpactModel->Excitable;
    if (Model == SoundObjectModel::Modal && ModalModel) return ModalModel->Excitable;

    static constexpr Excitable EmptyExcitable{};
    return EmptyExcitable;
}

Mesh2FaustResult GenerateDsp(const tetgenio &tets, const AcousticMaterialProperties &material, const std::vector<uint> &excitable_vertices, bool freq_control = false, std::optional<float> fundamental_freq_opt = {}) {
    std::vector<int> tet_indices;
    tet_indices.reserve(tets.numberoftetrahedra * 4 * 3); // 4 triangles per tetrahedron, 3 indices per triangle.
    // Turn each tetrahedron into 4 triangles.
    for (uint i = 0; i < uint(tets.numberoftetrahedra); ++i) {
        auto &result_indices = tets.tetrahedronlist;
        uint tri_i = i * 4;
        int a = result_indices[tri_i], b = result_indices[tri_i + 1], c = result_indices[tri_i + 2], d = result_indices[tri_i + 3];
        tet_indices.insert(tet_indices.end(), {a, b, c, d, a, b, c, d, a, b, c, d});
    }
    // Convert the tetrahedral mesh into a VegaFEM TetMesh.
    TetMesh volumetric_mesh{
        tets.numberofpoints, tets.pointlist, tets.numberoftetrahedra * 3, tet_indices.data(),
        material.YoungModulus, material.PoissonRatio, material.Density
    };

    static constexpr std::string model_name{"modalModel"};
    const auto m2f_result = m2f::mesh2faust(
        &volumetric_mesh,
        m2f::MaterialProperties{
            .youngModulus = material.YoungModulus,
            .poissonRatio = material.PoissonRatio,
            .density = material.Density,
            .alpha = material.Alpha,
            .beta = material.Beta
        },
        m2f::CommonArguments{
            .modelName = model_name,
            .freqControl = freq_control,
            .modesMinFreq = 20,
            // 20k is the upper limit of human hearing, but we often need to pitch down to match the
            // fundamental frequency of the true recording, so we double the upper limit.
            .modesMaxFreq = 40000,
            .targetNModes = 30, // number of synthesized modes, starting with the lowest frequency in the provided min/max range
            .femNModes = 80, // number of modes to be computed for the finite element analysis
            // Convert to signed ints.
            .exPos = excitable_vertices | transform([](uint i) { return int(i); }) | to<std::vector>(),
            .nExPos = int(excitable_vertices.size()),
            .debugMode = false,
        }
    );
    const std::string_view model_dsp = m2f_result.modelDsp;
    if (model_dsp.empty()) return {"process = 0;", {}, {}, {}, {{}}, {}};

    auto &mode_freqs = m2f_result.model.modeFreqs;
    const float fundamental_freq = fundamental_freq_opt ?
        *fundamental_freq_opt :
        !mode_freqs.empty() ? mode_freqs.front() :
                              440.0f;

    // Static code sections.
    static constexpr std::string to_sandh{" : ba.sAndH(gate);"}; // Add a sample and hold on the gate, in serial, and end the expression.
    static const std::string
        gain = "gain = hslider(\"Gain[scale:log]\",0.2,0,0.5,0.01);",
        t60_scale = "t60Scale = hslider(\"t60[scale:log][tooltip: Scale T60 decay values of all modes by the same amount.]\",1,0.1,10,0.01)" + to_sandh,
        gate = std::format("gate = button(\"{}[tooltip: When excitation source is 'Hammer', excites the vertex. With any excitation source, applies the current parameters.]\");", GateParamName),
        hammer_hardness = "hammerHardness = hslider(\"Hammer hardness[tooltip: Only has an effect when excitation source is 'Hammer'.]\",0.9,0,1,0.01)" + to_sandh,
        hammer_size = "hammerSize = hslider(\"Hammer size[tooltip: Only has an effect when excitation source is 'Hammer'.]\",0.1,0,1,0.01)" + to_sandh,
        hammer = "hammer(trig,hardness,size) = en.ar(att,att,trig)*no.noise : fi.lowpass(3,ctoff)\nwith{ ctoff = (1-size)*9500+500; att = (1-hardness)*0.01+0.001; };";

    // Variable code sections.
    const uint num_excite = excitable_vertices.size();
    const std::string
        freq = std::format("freq = hslider(\"Frequency[scale:log][tooltip: Fundamental frequency of the model]\",{},60,26000,1){}", fundamental_freq, to_sandh),
        ex_pos = std::format("exPos = nentry(\"{}\",{},0,{},1){}", ExciteIndexParamName, (num_excite - 1) / 2, num_excite - 1, to_sandh),
        modal_model = std::format("{}({}exPos,t60Scale)", model_name, freq_control ? "freq," : ""),
        process = std::format("process = hammer(gate,hammerHardness,hammerSize) : {}*gain;", modal_model);

    std::stringstream instrument;
    instrument << gate << '\n'
               << hammer_hardness << '\n'
               << hammer_size << '\n'
               << gain << '\n'
               << freq << '\n'
               << ex_pos << '\n'
               << t60_scale << '\n'
               << '\n'
               << hammer << '\n'
               << '\n'
               << process << '\n';

    return {
        .ModelDsp = std::format("{}{}", model_dsp, instrument.str()),
        .ModeFreqs = std::move(mode_freqs),
        .ModeT60s = std::move(m2f_result.model.modeT60s),
        .ModeGains = std::move(m2f_result.model.modeGains),
        .ExcitableVertices = std::move(excitable_vertices),
        .Material = std::move(material)
    };
}

void SoundObject::RenderControls(entt::registry &r, entt::entity entity) {
    if (auto &dsp_generator = DspGenerator) {
        if (auto m2f_result = dsp_generator->Render()) {
            dsp_generator.reset();
            Dsp.SetCode(m2f_result->ModelDsp);
            ModalModel = std::make_unique<ModalAudioModel>(Dsp, std::move(*m2f_result));
            SetModel(SoundObjectModel::Modal, r, entity);
        }
    }

    using namespace ImGui;

    if (ImpactModel) {
        PushID("AudioModel");
        auto model = int(Model);
        bool model_changed = RadioButton("Recordings", &model, int(SoundObjectModel::ImpactAudio));
        SameLine();
        model_changed |= RadioButton("Modal", &model, int(SoundObjectModel::Modal));
        PopID();
        if (model_changed) {
            SetModel(SoundObjectModel(model), r, entity);
        }
    } else if (Model == SoundObjectModel::ImpactAudio) {
        SetModel(SoundObjectModel::Modal, r, entity);
    }

    const bool impact_mode = Model == SoundObjectModel::ImpactAudio, modal_mode = Model == SoundObjectModel::Modal;
    if ((impact_mode && ImpactModel) || (modal_mode && ModalModel)) {
        const auto &excitable = GetExcitable();
        const auto &excitable_vertices = excitable.ExcitableVertices;
        const auto selected_vertex = excitable.SelectedVertex();
        if (BeginCombo("Vertex", std::to_string(selected_vertex).c_str())) {
            for (uint vertex : excitable_vertices) {
                if (Selectable(std::to_string(vertex).c_str(), vertex == selected_vertex)) {
                    SetVertex(vertex);
                    r.remove<ExcitedVertex>(entity);
                }
            }
            EndCombo();
        }
        const bool can_excite = (impact_mode && ImpactModel->CanExcite()) || (modal_mode && ModalModel->CanExcite());
        if (!can_excite) BeginDisabled();
        Button("Strike");
        if (IsItemActivated()) {
            r.emplace<ExcitedVertex>(entity, GetExcitable().SelectedVertex(), 1.f);
        } else if (IsItemDeactivated()) {
            r.remove<ExcitedVertex>(entity);
        }
        if (!can_excite) EndDisabled();
    }

    if (impact_mode) {
        if (ImpactModel) ImpactModel->Draw();
        return;
    }

    // Modal mode
    if (ModalModel) {
        ModalModel->Draw();
        if (ModalModel->Waveform && ImpactModel && ImpactModel->Waveform) {
            const auto &modal = *ModalModel->Waveform, &impact = *ImpactModel->Waveform;
            // const uint n_test_modes = std::min(ModalModel->ModeCount(), 10u);
            // Uncomment to cache `n_test_modes` peak frequencies for display in the spectrum plot.
            // modal.GetPeakFrequencies(n_test_modes);
            // impact.GetPeakFrequencies(n_test_modes);
            // RMSE is abyssmal in most cases...
            // const float rmse = RMSE(a->GetPeakFrequencies(n_test_modes), b->GetPeakFrequencies(n_test_modes));
            // Text("RMSE of top %d mode frequencies: %f", n_test_modes, rmse);
            SameLine();
            if (Button("Save wav files")) {
                const auto name = GetName(r, entity);
                // Save wav files for both the modal and real-world impact sounds.
                static const auto WavOutDir = fs::path{".."} / "audio_samples";
                modal.WriteWav(WavOutDir / std::format("{}-modal", name));
                impact.WriteWav(WavOutDir / std::format("{}-impact", name));
            }
        }
    }

    SeparatorText("Material properties");
    if (BeginCombo("Presets", Material.Name.c_str())) {
        for (const auto &material_choice : materials::acoustic::All) {
            const bool is_selected = (material_choice.Name == Material.Name);
            if (Selectable(material_choice.Name.c_str(), is_selected)) {
                Material = material_choice;
            }
            if (is_selected) SetItemDefaultFocus();
        }
        EndCombo();
    }

    auto &material_props = Material.Properties;
    Text("Density (kg/m^3)");
    InputDouble("##Density", &material_props.Density, 0.0f, 0.0f, "%.3f");
    Text("Young's modulus (Pa)");
    InputDouble("##Young's modulus", &material_props.YoungModulus, 0.0f, 0.0f, "%.3f");
    Text("Poisson's ratio");
    InputDouble("##Poisson's ratio", &material_props.PoissonRatio, 0.0f, 0.0f, "%.3f");
    Text("Rayleigh damping alpha/beta");
    InputDouble("##Rayleigh damping alpha", &material_props.Alpha, 0.0f, 0.0f, "%.3f");
    InputDouble("##Rayleigh damping beta", &material_props.Beta, 0.0f, 0.0f, "%.3f");
    const bool material_changed = ModalModel && material_props != ModalModel->M2F.Material;
    if (material_changed && Button("Reset material props")) material_props = ModalModel->M2F.Material;

    SeparatorText("Tet mesh");

    static bool quality_tets = false;
    Checkbox("Quality", &quality_tets);
    MeshEditor::HelpMarker("Add new Steiner points to the interior of the tet mesh to improve model quality.");

    SeparatorText("Excitable vertices");
    static bool vertices_changed{false};
    // If impact model is present, default the modal model to be excitable at exactly the same points.
    static bool use_impact_vertices{ImpactModel};
    if (ImpactModel) vertices_changed = Checkbox("Use RealImpact vertices", &use_impact_vertices);
    else use_impact_vertices = false;

    const auto &mesh = r.get<const Mesh>(entity);
    static int num_excitable_vertices = 10;
    if (!use_impact_vertices) {
        const auto num_points = mesh.GetVertexCount();
        if (uint(num_excitable_vertices) > num_points) num_excitable_vertices = num_points;
        vertices_changed |= SliderInt("Num excitable vertices", &num_excitable_vertices, 1, num_points);
    }

    const auto num_points = mesh.GetVertexCount();
    static std::vector<uint> excitable_vertices;
    if (excitable_vertices.empty() || vertices_changed) {
        // Use impact model vertices or linearly distribute the vertices across the tet mesh.
        excitable_vertices = use_impact_vertices ?
            ImpactModel->Excitable.ExcitableVertices :
            iota_view{0u, uint(num_excitable_vertices)} | transform([&](uint i) { return i * num_points / num_excitable_vertices; }) | to<std::vector<uint>>();
    }

    const bool disable_generate = !material_changed && ModalModel && excitable_vertices == ModalModel->Excitable.ExcitableVertices;
    if (disable_generate) BeginDisabled();
    if (Button(std::format("{} audio model", ModalModel ? "Regenerate" : "Generate").c_str())) {
        const auto scale = r.get<Scale>(entity).Value;
        const auto fundamental_freq = ImpactModel && ImpactModel->Waveform ? std::optional{ImpactModel->Waveform->GetPeakFrequencies(8).front()} : std::nullopt;
        DspGenerator = std::make_unique<Worker<Mesh2FaustResult>>("Generating modal audio model...", [&, scale, fundamental_freq] {
            // todo Add an invisible tet mesh to the scene and support toggling between surface/volumetric tet mesh views.
            // scene.AddMesh(tets->CreateMesh(), {.Name = "Tet Mesh", R.get<Model>(selected_entity).Transform;, .Select = false, .Visible = false});

            // We rely on `PreserveSurface` behavior for excitable vertices;
            // Vertex indices on the surface mesh must match vertex indices on the tet mesh.
            // todo display tet mesh in UI and select vertices for debugging (just like other meshes but restrict to edge view)

            while (!DspGenerator) {}
            DspGenerator->SetMessage("Generating tetrahedral mesh...");
            const auto tets = GenerateTets(mesh, scale, {.PreserveSurface = true, .Quality = quality_tets});

            DspGenerator->SetMessage("Generating DSP...");
            return ::GenerateDsp(*tets, material_props, excitable_vertices, true, fundamental_freq);
        });
    }
    if (disable_generate) EndDisabled();
}
