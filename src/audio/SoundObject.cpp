#include "SoundObject.h"

#include <format>
#include <print>
#include <ranges>

#include "imgui.h"
#include "implot.h"
#include "miniaudio.h"

#include "mesh2faust.h"

using Sample = float;
#ifndef FAUSTFLOAT
#define FAUSTFLOAT Sample
#endif

#include "draw/drawschema.hh" // faust/compiler/draw/drawschema.hh
#include "faust/dsp/llvm-dsp.h"

#include "FFTData.h"
#include "FaustParams.h"

#include "tetMesh.h" // Vega
#include "tetgen.h" // Must be after any Faust includes, since it defined a `REAL` macro.

#include "Tets.h"
#include "Worker.h"

using std::ranges::iota_view;
using std::ranges::to;
using std::views::transform;

struct Mesh2FaustResult {
    std::string ModelDsp; // Faust DSP code defining the model function.
    std::vector<float> ModeFreqs; // Mode frequencies
    std::vector<float> ModeT60s; // Mode T60 decay times
    std::vector<std::vector<float>> ModeGains; // Mode gains by [exitation position][mode]
    std::vector<uint> ExcitableVertices; // Copy of the excitable vertices used for model generation.
};

namespace {
struct ImpactRecording {
    static constexpr uint FrameCount = 208592; // Same length as RealImpact recordings.
    float Frames[FrameCount];
    uint CurrentFrame{0};
    bool Complete{false};
};

constexpr uint SampleRate = 48000; // todo respect device sample rate

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

    std::vector<std::pair<float, uint>> peaks;
    for (uint i = 1; i < N_2 - 1; i++) {
        float mag_sq = data[i][0] * data[i][0] + data[i][1] * data[i][1];
        float left_mag_sq = data[i - 1][0] * data[i - 1][0] + data[i - 1][1] * data[i - 1][1];
        float right_mag_sq = data[i + 1][0] * data[i + 1][0] + data[i + 1][1] * data[i + 1][1];
        if (mag_sq > left_mag_sq && mag_sq > right_mag_sq) peaks.emplace_back(mag_sq, i);
    }

    // Sort descending by magnitude and convert to frequency.
    std::ranges::sort(peaks, [](const auto &a, const auto &b) { return a.first > b.first; });
    auto peak_freqs = peaks | std::views::take(n_peaks) |
        transform([n_bins](const auto &p) { return float(p.second * SampleRate / n_bins); }) | to<std::vector>();
    std::ranges::sort(peak_freqs);
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
    std::vector<float> PeakFrequencies;

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

    std::vector<float> GetPeakFrequencies(uint n_peaks) {
        if (n_peaks != PeakFrequencies.size()) PeakFrequencies = ::FindPeakFrequencies(FftData.Complex, WindowedFrames.size(), n_peaks);
        return PeakFrequencies;
    }

    float GetMaxValue() const { return *std::max_element(Frames.begin(), Frames.end()); }

    // If `normalize_max` is set, normalize the data to this maximum value.
    void WriteWav(std::string_view file_name, std::optional<float> normalize_max = std::nullopt) const {
        const std::string wav_filename = std::format("../audio_samples/{}.wav", file_name);
        if (auto status = ma_encoder_init_file(wav_filename.c_str(), &WavEncoderConfig, &WavEncoder); status != MA_SUCCESS) {
            throw std::runtime_error(std::format("Failed to initialize wav file {}. Status: {}", wav_filename, uint(status)));
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
    ImpactAudioModel(std::unordered_map<uint, std::vector<float>> &&impact_frames_by_vertex, uint vertex)
        : ImpactFramesByVertex(std::move(impact_frames_by_vertex)),
          // All samples are the same length.
          MaxFrame(ImpactFramesByVertex.empty() ? 0 : ImpactFramesByVertex.begin()->second.size()) {
        SetVertex(vertex);
    }
    ~ImpactAudioModel() = default;

    const ImpactAudioModel &operator=(ImpactAudioModel &&other) noexcept {
        if (this != &other) {
            ImpactFramesByVertex = std::move(other.ImpactFramesByVertex);
            CurrentFrame = other.CurrentFrame;
            Waveform = std::move(other.Waveform);
        }
        return *this;
    }

    std::unordered_map<uint, std::vector<float>> ImpactFramesByVertex;
    uint MaxFrame;
    uint CurrentFrame{MaxFrame}; // Start at the end, so it doesn't immediately play.
    std::unique_ptr<Waveform> Waveform; // Current vertex's waveform

    void Start() { CurrentFrame = 0; }
    void Stop() { CurrentFrame = MaxFrame; }
    bool IsStarted() const { return CurrentFrame != MaxFrame; }

    bool CanStrike() const { return bool(Waveform); }
    void SetVertex(uint vertex) {
        Stop();
        if (ImpactFramesByVertex.contains(vertex)) {
            auto &frames = ImpactFramesByVertex.at(vertex);
            Waveform = std::make_unique<::Waveform>(frames.data(), frames.size());
        }
    }

    void SetVertexForce(float force) {
        if (force > 0 && !IsStarted()) Start();
        else if (force == 0 && IsStarted()) Stop();
    }

    void Draw() const {
        if (!Waveform) return;

        Waveform->PlotFrames("Real-world impact waveform", CurrentFrame);
        Waveform->PlotMagnitudeSpectrum("Real-world impact spectrum");
    }
};

namespace {
// `FaustDSP` is a wrapper around a Faust DSP and Box.
// It has a Faust DSP code string, and updates its DSP and Box instances to reflect the current code.
struct FaustDSP {
    FaustDSP(std::string_view code) {
        SetCode(std::move(code));
    }
    ~FaustDSP() {
        Uninit();
    }

    Box Box{nullptr};
    dsp *Dsp{nullptr};
    std::unique_ptr<FaustParams> Params;

    std::string ErrorMessage{""};

    void SetCode(std::string_view code) {
        Code = std::move(code);
        Update();
    }
    std::string GetCode() const { return Code; }

    void Compute(uint n, Sample **input, Sample **output) {
        if (Dsp != nullptr) Dsp->compute(n, input, output);
    }

    void DrawParams() {
        if (Params) Params->Draw();
    }

    Sample Get(std::string_view param_label) {
        if (auto *zone = GetZone(param_label)) return *zone;
        return 0;
    }

    void Set(std::string_view param_label, Sample value) {
        if (auto *zone = GetZone(param_label); zone) *zone = value;
    }

    Sample *GetZone(std::string_view param_label) { return Params ? Params->getZoneForLabel(param_label.data()) : nullptr; }

    void SaveSvg() {
        drawSchema(Box, "MeshEditor-svg", "svg");
    }

private:
    std::string Code{""};
    llvm_dsp_factory *DspFactory{nullptr};

    void Init() {
        if (Code.empty()) return;

        createLibContext();

        static constexpr std::string AppName{"MeshEditor"};
        static const std::string LibrariesPath{fs::relative("../lib/faust/libraries")};
        std::vector<const char *> argv = {"-I", LibrariesPath.c_str()};
        if (std::is_same_v<Sample, double>) argv.push_back("-double");
        const int argc = argv.size();

        static int num_inputs, num_outputs;
        Box = DSPToBoxes(AppName, Code, argc, argv.data(), &num_inputs, &num_outputs, ErrorMessage);

        if (Box && ErrorMessage.empty()) {
            static constexpr int optimize_level = -1;
            DspFactory = createDSPFactoryFromBoxes(AppName, Box, argc, argv.data(), "", ErrorMessage, optimize_level);
            if (DspFactory) {
                if (ErrorMessage.empty()) {
                    Dsp = DspFactory->createDSPInstance();
                    if (!Dsp) ErrorMessage = "Successfully created Faust DSP factory, but could not create the Faust DSP instance.";

                    Dsp->init(SampleRate); // todo follow device sample rate
                    Params = std::make_unique<FaustParams>();
                    Dsp->buildUserInterface(Params.get());
                } else {
                    deleteDSPFactory(DspFactory);
                    DspFactory = nullptr;
                }
            }
        } else if (!Box && ErrorMessage.empty()) {
            ErrorMessage = "`DSPToBoxes` returned no error but did not produce a result.";
        }
    }
    void Uninit() {
        if (Dsp || DspFactory) DestroyDsp();
        if (Box) Box = nullptr;
        ErrorMessage = "";
        destroyLibContext();
    }

    void Update() {
        Uninit();
        Init();
    }

    void DestroyDsp() {
        Params.reset();
        if (Dsp) {
            delete Dsp;
            Dsp = nullptr;
        }
        if (DspFactory) {
            deleteDSPFactory(DspFactory);
            DspFactory = nullptr;
        }
    }
};

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

constexpr std::string ExciteIndexParamName{"Excite index"};
constexpr std::string GateParamName{"Gate"};
} // namespace

struct ModalAudioModel {
    ModalAudioModel(Mesh2FaustResult &&m2f, uint vertex) : ExcitableVertices(m2f.ExcitableVertices) {
        FaustDsp = std::make_unique<FaustDSP>(m2f.ModelDsp);
        ModeFreqs = std::move(m2f.ModeFreqs);
        ModeT60s = std::move(m2f.ModeT60s);
        ModeGains = std::move(m2f.ModeGains);
        SetVertex(vertex);
    }

    ~ModalAudioModel() = default;

    uint ModeCount() const { return ModeFreqs.size(); }

    void ProduceAudio(float *input, float *output, uint frame_count) const {
        if (ImpactRecording && ImpactRecording->CurrentFrame == 0) SetParam(GateParamName, 1);

        if (FaustDsp) FaustDsp->Compute(frame_count, &input, &output);

        if (ImpactRecording && !ImpactRecording->Complete) {
            for (uint i = 0; i < frame_count && ImpactRecording->CurrentFrame < ImpactRecording::FrameCount; ++i, ++ImpactRecording->CurrentFrame) {
                ImpactRecording->Frames[ImpactRecording->CurrentFrame] = output[i];
            }
            if (ImpactRecording->CurrentFrame == ImpactRecording::FrameCount) {
                ImpactRecording->Complete = true;
                SetParam(GateParamName, 0);
            }
        }
    }

    bool CanStrike() const { return FaustDsp && (!ImpactRecording || ImpactRecording->Complete); }
    void SetVertex(uint vertex) {
        Stop();
        if (auto it = std::ranges::find(ExcitableVertices, vertex); it != ExcitableVertices.end()) {
            SetParam(ExciteIndexParamName, std::ranges::distance(ExcitableVertices.begin(), it));
        }
    }
    void SetVertexForce(float force) { SetParam(GateParamName, force); }
    void Stop() { SetVertexForce(0); }

    void SetParam(std::string_view param_label, Sample param_value) const {
        if (FaustDsp) FaustDsp->Set(std::move(param_label), param_value);
    }

    void Draw(uint *selected_vertex_index) {
        using namespace ImGui;

        if (!FaustDsp) return;

        if (Button("Save DSP SVG")) FaustDsp->SaveSvg();

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
            Waveform->PlotMagnitudeSpectrum("Modal impact spectrum", HoveredModeIndex ? std::optional{*HoveredModeIndex} : std::nullopt);
        }

        auto &dsp = *FaustDsp;
        // Poll the Faust DSP UI to see if the current excitation vertex has changed.
        const auto vertex_index = uint(dsp.Get(ExciteIndexParamName));
        if (vertex_index < ExcitableVertices.size() && selected_vertex_index != nullptr) {
            *selected_vertex_index = ExcitableVertices[vertex_index];
        }

        if (CollapsingHeader("Modal data charts")) {
            std::optional<size_t> new_hovered_index;
            if (auto hovered = PlotModeData(ModeFreqs, "Mode frequencies", "", "Frequency (Hz)", HoveredModeIndex)) new_hovered_index = hovered;
            if (auto hovered = PlotModeData(ModeT60s, "Mode T60s", "", "T60 decay time (s)", HoveredModeIndex)) new_hovered_index = hovered;
            if (auto hovered = PlotModeData(ModeGains[vertex_index], "Mode gains", "Mode index", "Gain", HoveredModeIndex, 1.f)) new_hovered_index = hovered;
            HoveredModeIndex = new_hovered_index;
            if (HoveredModeIndex && *HoveredModeIndex < ModeFreqs.size()) {
                const auto hovered_index = *HoveredModeIndex;
                Text(
                    "Mode %lu: Freq %.2f Hz, T60 %.2f s, Gain %.2f dB", hovered_index,
                    ModeFreqs[hovered_index],
                    ModeT60s[hovered_index],
                    ModeGains[vertex_index][hovered_index]
                );
            }
        }

        SeparatorText("DSP");
        if (Button("Print DSP code")) std::println("DSP code:\n\n{}\n", dsp.GetCode());
        dsp.DrawParams();
    }

    std::unique_ptr<Waveform> Waveform; // Recorded waveform

private:
    std::vector<uint> ExcitableVertices;

    // todo use Mesh2FaustResult
    std::unique_ptr<FaustDSP> FaustDsp;
    std::vector<float> ModeFreqs{};
    std::vector<float> ModeT60s{};
    std::vector<std::vector<float>> ModeGains{};

    std::unique_ptr<ImpactRecording> ImpactRecording;
    std::optional<size_t> HoveredModeIndex;
};

namespace {
Mesh2FaustResult GenerateDsp(const tetgenio &tets, const MaterialProperties &material, const std::vector<uint> &excitable_vertex_indices, bool freq_control = false, std::optional<float> fundamental_freq_opt = std::nullopt) {
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
            .exPos = excitable_vertex_indices | transform([](uint i) { return int(i); }) | to<std::vector>(),
            .nExPos = int(excitable_vertex_indices.size()),
            .debugMode = false,
        }
    );
    const std::string model_dsp = m2f_result.modelDsp;
    if (model_dsp.empty()) return {"process = 0;", {}, {}, {}, {}};

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
    const uint num_excite_pos = excitable_vertex_indices.size();
    const std::string
        freq = std::format("freq = hslider(\"Frequency[scale:log][tooltip: Fundamental frequency of the model]\",{},60,26000,1){}", fundamental_freq, to_sandh),
        ex_pos = std::format("exPos = nentry(\"{}\",{},0,{},1){}", ExciteIndexParamName, (num_excite_pos - 1) / 2, num_excite_pos - 1, to_sandh),
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
        .ModelDsp = model_dsp + instrument.str(),
        .ModeFreqs = std::move(mode_freqs),
        .ModeT60s = std::move(m2f_result.model.modeT60s),
        .ModeGains = std::move(m2f_result.model.modeGains),
        .ExcitableVertices = excitable_vertex_indices
    };
}

MaterialProperties GetMaterialPreset(std::string_view name) {
    if (MaterialPresets.contains(name)) return MaterialPresets.at(name);
    return MaterialPresets.at(DefaultMaterialPresetName);
}

// Assumes a and b are the same length.
constexpr float RMSE(const std::vector<float> &a, const std::vector<float> &b) {
    float sum = 0;
    for (size_t i = 0; i < a.size(); ++i) sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrtf(sum / a.size());
}
} // namespace

SoundObject::SoundObject(std::string_view name, const ::Tets &tets, const std::optional<std::string_view> &material_name)
    : Name(name), Tets(tets), MaterialName(material_name.value_or(DefaultMaterialPresetName)), Material(GetMaterialPreset(MaterialName)) {}

SoundObject::~SoundObject() = default;

void SoundObject::SetImpactFrames(std::unordered_map<uint, std::vector<float>> &&impact_frames_by_vertex) {
    ExcitableVertices.clear();
    if (!impact_frames_by_vertex.empty()) {
        for (auto &[vertex, _] : impact_frames_by_vertex) ExcitableVertices.emplace_back(vertex);
        CurrentVertex = ExcitableVertices.front();
        ImpactModel = std::make_unique<ImpactAudioModel>(std::move(impact_frames_by_vertex), CurrentVertex);
    }
}

void SoundObject::ProduceAudio(DeviceData device, float *input, float *output, uint frame_count) {
    if (Model == SoundObjectModel::ImpactAudio && ImpactModel) {
        if (!ImpactModel->ImpactFramesByVertex.contains(CurrentVertex)) return;

        const auto &impact_samples = ImpactModel->ImpactFramesByVertex.at(CurrentVertex);
        const uint sample_rate = device.SampleRate; // todo - resample from 48kHz to device sample rate if necessary
        (void)sample_rate; // Unused

        for (uint i = 0; i < frame_count; ++i) {
            output[i] += ImpactModel->CurrentFrame < impact_samples.size() ? impact_samples[ImpactModel->CurrentFrame++] : 0.0f;
        }
    } else if (Model == SoundObjectModel::Modal && ModalModel) {
        ModalModel->ProduceAudio(input, output, frame_count);
    }
}

void SoundObject::SetModel(SoundObjectModel model) {
    // Stop any ongoing impacts.
    if (ImpactModel) ImpactModel->Stop();
    if (ModalModel) ModalModel->Stop();
    Model = model;
}

std::optional<uint> SoundObject::FindNearestExcitableVertex(vec3 position) {
    if (ExcitableVertices.empty()) return {};

    std::optional<uint> nearest_excite_vertex{};
    float min_dist = FLT_MAX;
    for (uint excite_vertex : ExcitableVertices) {
        const float dist = glm::distance(position, Tets.GetVertexPosition(excite_vertex));
        if (dist < min_dist) {
            min_dist = dist;
            nearest_excite_vertex = {excite_vertex};
        }
    }
    return nearest_excite_vertex;
}

void SoundObject::SetVertex(uint vertex) {
    if (CurrentVertex == vertex) return;

    CurrentVertex = vertex;
    // Update vertex in all present models.
    if (ImpactModel) ImpactModel->SetVertex(CurrentVertex);
    if (ModalModel) ModalModel->SetVertex(CurrentVertex);
}

void SoundObject::SetVertexForce(float force) {
    // Update vertex force in the active model.
    if (Model == SoundObjectModel::ImpactAudio && ImpactModel) ImpactModel->SetVertexForce(force);
    else if (Model == SoundObjectModel::Modal && ModalModel) ModalModel->SetVertexForce(force);
}

void SoundObject::RenderControls() {
    using namespace ImGui;

    if (ImpactModel) {
        PushID("AudioModel");
        int model = int(Model);
        bool model_changed = RadioButton("Recordings", &model, int(SoundObjectModel::ImpactAudio));
        SameLine();
        model_changed |= RadioButton("Modal", &model, int(SoundObjectModel::Modal));
        PopID();
        if (model_changed) SetModel(SoundObjectModel(model));
    } else {
        Model = SoundObjectModel::Modal;
    }

    const bool impact_mode = Model == SoundObjectModel::ImpactAudio, modal_mode = Model == SoundObjectModel::Modal;
    const bool model_present = (impact_mode && ImpactModel) || (modal_mode && ModalModel);
    if (model_present) {
        if (BeginCombo("Vertex", std::to_string(CurrentVertex).c_str())) {
            for (uint vertex : ExcitableVertices) {
                if (Selectable(std::to_string(vertex).c_str(), vertex == CurrentVertex)) {
                    SetVertex(vertex);
                }
            }
            EndCombo();
        }
        const bool can_strike = (impact_mode && ImpactModel->CanStrike()) || (modal_mode && ModalModel->CanStrike());
        if (!can_strike) BeginDisabled();
        Button("Strike");
        if (IsItemActivated()) SetVertexForce(1);
        else if (IsItemDeactivated()) SetVertexForce(0);
        if (!can_strike) EndDisabled();
    }
    if (impact_mode && model_present) {
        ImpactModel->Draw();
    } else if (modal_mode) {
        if (model_present) {
            if (ModalModel->Waveform && ImpactModel && ImpactModel->Waveform) {
                auto &modal = *ModalModel->Waveform, &impact = *ImpactModel->Waveform;
                // const uint n_test_modes = std::min(ModalModel->ModeCount(), 10u);
                // Uncomment to cache `n_test_modes` peak frequencies for display in the spectrum plot.
                // modal.GetPeakFrequencies(n_test_modes);
                // impact.GetPeakFrequencies(n_test_modes);
                // RMSE is abyssmal in most cases...
                // const float rmse = RMSE(a->GetPeakFrequencies(n_test_modes), b->GetPeakFrequencies(n_test_modes));
                // Text("RMSE of top %d mode frequencies: %f", n_test_modes, rmse);
                SameLine();
                if (Button("Save wav files")) {
                    // Save wav files for both the modal and real-world impact sounds.
                    modal.WriteWav(std::format("{}-modal", Name, impact.GetMaxValue()));
                    impact.WriteWav(std::format("{}-impact", Name));
                }
            }
            ModalModel->Draw(&CurrentVertex);
        }

        SeparatorText("Material properties");
        if (BeginCombo("Presets", MaterialName.data())) {
            for (const auto [preset_name, material] : MaterialPresets) {
                const bool is_selected = (preset_name == MaterialName);
                if (Selectable(preset_name.data(), is_selected)) {
                    MaterialName = preset_name;
                    Material = material;
                }
                if (is_selected) SetItemDefaultFocus();
            }
            EndCombo();
        }

        Text("Density (kg/m^3)");
        InputDouble("##Density", &Material.Density, 0.0f, 0.0f, "%.3f");
        Text("Young's modulus (Pa)");
        InputDouble("##Young's modulus", &Material.YoungModulus, 0.0f, 0.0f, "%.3f");
        Text("Poisson's ratio");
        InputDouble("##Poisson's ratio", &Material.PoissonRatio, 0.0f, 0.0f, "%.3f");
        Text("Rayleigh damping alpha/beta");
        InputDouble("##Rayleigh damping alpha", &Material.Alpha, 0.0f, 0.0f, "%.3f");
        InputDouble("##Rayleigh damping beta", &Material.Beta, 0.0f, 0.0f, "%.3f");
        if (DspGenerator) {
            if (auto m2f_result = DspGenerator->Render()) {
                ModalModel = std::make_unique<ModalAudioModel>(std::move(*m2f_result), CurrentVertex);
                DspGenerator.reset();
            }
        }
        if (Button(std::format("{} DSP", ModalModel ? "Regenerate" : "Generate").c_str())) {
            if (!ImpactModel) {
                // ImpactAudio objects can only be struck at the impact points.
                // Otherwise, linearly distribute the vertices across the tet mesh.
                const uint num_excitable_vertices = 5; // todo UI input
                ExcitableVertices = iota_view{0u, num_excitable_vertices} | transform([&](uint i) { return i * Tets->numberofpoints / num_excitable_vertices; }) | to<std::vector>();
            }
            std::optional<float> fundamental_freq = ImpactModel && ImpactModel->Waveform ? std::optional{ImpactModel->Waveform->GetPeakFrequencies(10).front()} : std::nullopt;
            DspGenerator = std::make_unique<Worker<Mesh2FaustResult>>("Generating DSP code...", [&] {
                return GenerateDsp(*Tets, Material, ExcitableVertices, true, fundamental_freq);
            });
        }
    }
}
