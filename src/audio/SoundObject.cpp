#include "SoundObject.h"

#include <format>
#include <print>
#include <ranges>

#include "imgui.h"
#include "implot.h"
#include "miniaudio.h"

using Sample = float;
#ifndef FAUSTFLOAT
#define FAUSTFLOAT Sample
#endif

#include "mesh2faust.h"

#include "draw/drawschema.hh" // faust/compiler/draw/drawschema.hh
#include "faust/dsp/llvm-dsp.h"

#include "FaustParams.h"

#include "tetMesh.h" // Vega
#include "tetgen.h" // Must be after any Faust includes, since it defined a `REAL` macro.

#include "AcousticMaterial.h"
#include "AudioBuffer.h"
#include "FFTData.h"
#include "SvgResource.h"
#include "Tets.h"
#include "Worker.h"

using std::ranges::find, std::ranges::iota_view, std::ranges::sort, std::ranges::to;
using std::views::transform;

struct Mesh2FaustResult {
    std::string ModelDsp; // Faust DSP code defining the model function.
    std::vector<float> ModeFreqs; // Mode frequencies
    std::vector<float> ModeT60s; // Mode T60 decay times
    std::vector<std::vector<float>> ModeGains; // Mode gains by [exitation position][mode]
    std::vector<uint> ExcitableVertices; // Excitable vertices
};

namespace {
struct ImpactRecording {
    static constexpr uint FrameCount = 208592; // Same length as RealImpact recordings.
    float Frames[FrameCount];
    uint Frame{0};
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
    sort(peaks, [](const auto &a, const auto &b) { return a.first > b.first; });
    auto peak_freqs = peaks | std::views::take(n_peaks) |
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

    std::vector<float> GetPeakFrequencies(uint n_peaks) {
        if (n_peaks != PeakFrequencies.size()) PeakFrequencies = ::FindPeakFrequencies(FftData.Complex, WindowedFrames.size(), n_peaks);
        return PeakFrequencies;
    }

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
    ImpactAudioModel(std::unordered_map<uint, std::vector<float>> &&impact_frames_by_vertex)
        : ImpactFramesByVertex(std::move(impact_frames_by_vertex)),
          Excitable(ImpactFramesByVertex | std::views::keys | to<std::vector>()),
          // All samples are the same length.
          MaxFrame(ImpactFramesByVertex.empty() ? 0 : ImpactFramesByVertex.begin()->second.size()) {
        SetVertex(Excitable.SelectedVertex);
    }
    ~ImpactAudioModel() = default;

    const ImpactAudioModel &operator=(ImpactAudioModel &&other) noexcept {
        if (this != &other) {
            ImpactFramesByVertex = std::move(other.ImpactFramesByVertex);
            Frame = other.Frame;
            Waveform = std::move(other.Waveform);
        }
        return *this;
    }

    std::unordered_map<uint, std::vector<float>> ImpactFramesByVertex;
    Excitable Excitable;
    uint MaxFrame;
    uint Frame{MaxFrame}; // Start at the end, so it doesn't immediately play.
    std::unique_ptr<Waveform> Waveform; // Selected vertex's waveform

    void ProduceAudio(const AudioBuffer &buffer) {
        if (ImpactFramesByVertex.empty()) return;

        const auto &impact_samples = ImpactFramesByVertex.at(Excitable.SelectedVertex);
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
        Stop();
        Excitable.SelectedVertex = vertex;
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

        Waveform->PlotFrames("Real-world impact waveform", Frame);
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

    void Compute(uint n, const Sample **input, Sample **output) const {
        if (Dsp) Dsp->compute(n, const_cast<Sample **>(input), output);
    }

    void DrawParams() {
        if (Params) Params->Draw();
    }

    Sample Get(std::string_view param_label) {
        if (auto *zone = GetZone(param_label)) return *zone;
        return 0;
    }

    void Set(std::string_view param_label, Sample value) const {
        if (auto *zone = GetZone(param_label); zone) *zone = value;
    }

    Sample *GetZone(std::string_view param_label) const {
        return Params ? Params->getZoneForLabel(param_label.data()) : nullptr;
    }

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
    ModalAudioModel(Mesh2FaustResult &&m2f, CreateSvgResource create_svg)
        : FaustDsp(m2f.ModelDsp), M2F(std::move(m2f)), Excitable(M2F.ExcitableVertices), CreateSvg(std::move(create_svg)) {
        M2F.ExcitableVertices.clear(); // These are only used to populate `Excitable`.
        SetVertex(Excitable.SelectedVertex);
    }

    ~ModalAudioModel() = default;

    uint ModeCount() const { return M2F.ModeFreqs.size(); }

    void ProduceAudio(AudioBuffer &buffer) const {
        if (ImpactRecording && ImpactRecording->Frame == 0) SetParam(GateParamName, 1);

        FaustDsp.Compute(buffer.FrameCount, &buffer.Input, &buffer.Output);

        if (ImpactRecording && !ImpactRecording->Complete) {
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
        Excitable.SelectedVertex = vertex;
        const auto &vertices = Excitable.ExcitableVertices;
        if (auto it = find(vertices, vertex); it != vertices.end()) {
            SetParam(ExciteIndexParamName, std::ranges::distance(vertices.begin(), it));
        }
    }
    void SetVertexForce(float force) { SetParam(GateParamName, force); }
    void Stop() { SetVertexForce(0); }

    void SetParam(std::string_view param_label, Sample param_value) const {
        FaustDsp.Set(std::move(param_label), param_value);
    }

    void DrawDspGraph() {
        const static fs::path FaustSvgDir = "MeshEditor-svg";
        if (!fs::exists(FaustSvgDir)) FaustDsp.SaveSvg();

        static fs::path SelectedSvg = "process.svg";
        if (const auto faust_svg_path = FaustSvgDir / SelectedSvg; fs::exists(faust_svg_path)) {
            if (!FaustSvg || FaustSvg->Path != faust_svg_path) {
                CreateSvg(FaustSvg, faust_svg_path);
            }
            if (auto clickedLinkOpt = FaustSvg->Render()) {
                SelectedSvg = std::move(*clickedLinkOpt);
            }
        }
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
        auto selected_vertex_index = uint(FaustDsp.Get(ExciteIndexParamName));
        const auto &vertices = Excitable.ExcitableVertices;
        Excitable.SelectedVertex = selected_vertex_index < vertices.size() ? vertices[selected_vertex_index] : 0;
        if (CollapsingHeader("Modal data charts")) {
            std::optional<size_t> new_hovered_index;
            if (auto hovered = PlotModeData(M2F.ModeFreqs, "Mode frequencies", "", "Frequency (Hz)", HoveredModeIndex)) new_hovered_index = hovered;
            if (auto hovered = PlotModeData(M2F.ModeT60s, "Mode T60s", "", "T60 decay time (s)", HoveredModeIndex)) new_hovered_index = hovered;
            if (auto hovered = PlotModeData(M2F.ModeGains[selected_vertex_index], "Mode gains", "Mode index", "Gain", HoveredModeIndex, 1.f)) new_hovered_index = hovered;
            if (HoveredModeIndex = new_hovered_index; HoveredModeIndex && *HoveredModeIndex < M2F.ModeFreqs.size()) {
                const auto index = *HoveredModeIndex;
                Text(
                    "Mode %lu: Freq %.2f Hz, T60 %.2f s, Gain %.2f dB", index,
                    M2F.ModeFreqs[index],
                    M2F.ModeT60s[index],
                    M2F.ModeGains[selected_vertex_index][index]
                );
            }
        }

        if (CollapsingHeader("DSP parameters")) FaustDsp.DrawParams();
        if (CollapsingHeader("DSP graph")) DrawDspGraph();
        if (Button("Print DSP code")) std::println("DSP code:\n\n{}\n", FaustDsp.GetCode());
    }

    std::unique_ptr<Waveform> Waveform{}; // Recorded waveform

    FaustDSP FaustDsp;
    Mesh2FaustResult M2F;
    Excitable Excitable;

private:
    CreateSvgResource CreateSvg;

    std::unique_ptr<SvgResource> FaustSvg;
    std::unique_ptr<ImpactRecording> ImpactRecording;
    std::optional<size_t> HoveredModeIndex;
};

namespace {
Mesh2FaustResult GenerateDsp(const tetgenio &tets, const AcousticMaterialProperties &material, std::vector<uint> &&excitable_vertices, bool freq_control = false, std::optional<float> fundamental_freq_opt = std::nullopt) {
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
    const std::string model_dsp = m2f_result.modelDsp;
    if (model_dsp.empty()) return {"process = 0;", {}, {}, {}, {{}}};

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
        .ModelDsp = model_dsp + instrument.str(),
        .ModeFreqs = std::move(mode_freqs),
        .ModeT60s = std::move(m2f_result.model.modeT60s),
        .ModeGains = std::move(m2f_result.model.modeGains),
        .ExcitableVertices = std::move(excitable_vertices),
    };
}

// Assumes a and b are the same length.
constexpr float RMSE(const std::vector<float> &a, const std::vector<float> &b) {
    float sum = 0;
    for (size_t i = 0; i < a.size(); ++i) sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrtf(sum / a.size());
}
} // namespace

SoundObject::SoundObject(CreateSvgResource create_svg) : CreateSvg(std::move(create_svg)) {}

SoundObject::~SoundObject() = default;

void SoundObject::Apply(SoundObjectAction::Any action) {
    std::visit(
        Match{
            [&](SoundObjectAction::SetModel action) {
                if (ImpactModel) ImpactModel->Stop();
                if (ModalModel) ModalModel->Stop();
                Model = action.Model;
            },
            [&](SoundObjectAction::SelectVertex action) {
                SetVertex(action.Vertex);
            },
            [&](SoundObjectAction::Excite action) {
                SetVertex(action.Vertex);
                SetVertexForce(action.Force);
            },
            [&](SoundObjectAction::SetExciteForce action) {
                SetVertexForce(action.Force);
            }
        },
        std::move(action)
    );
}

void SoundObject::SetImpactFrames(std::unordered_map<uint, std::vector<float>> &&impact_frames_by_vertex) {
    if (!impact_frames_by_vertex.empty()) {
        ImpactModel = std::make_unique<ImpactAudioModel>(std::move(impact_frames_by_vertex));
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

const Excitable &SoundObject::GetExcitable() const {
    if (Model == SoundObjectModel::ImpactAudio && ImpactModel) return ImpactModel->Excitable;
    if (Model == SoundObjectModel::Modal && ModalModel) return ModalModel->Excitable;

    static constexpr Excitable EmptyExcitable{};
    return EmptyExcitable;
}

std::optional<SoundObjectAction::Any> SoundObject::RenderControls(std::string_view name, const Tets &tets, AcousticMaterial &material) {
    using namespace ImGui;

    std::optional<SoundObjectAction::Any> action;

    if (ImpactModel) {
        PushID("AudioModel");
        auto model = int(Model);
        bool model_changed = RadioButton("Recordings", &model, int(SoundObjectModel::ImpactAudio));
        SameLine();
        model_changed |= RadioButton("Modal", &model, int(SoundObjectModel::Modal));
        PopID();
        if (model_changed) action = SoundObjectAction::SetModel{SoundObjectModel(model)};
    } else if (Model == SoundObjectModel::ImpactAudio) {
        action = SoundObjectAction::SetModel{SoundObjectModel::Modal};
    }

    const bool impact_mode = Model == SoundObjectModel::ImpactAudio, modal_mode = Model == SoundObjectModel::Modal;
    if ((impact_mode && ImpactModel) || (modal_mode && ModalModel)) {
        const auto &excitable = GetExcitable();
        const auto &excitable_vertices = excitable.ExcitableVertices;
        const auto selected_vertex = excitable.SelectedVertex;
        if (BeginCombo("Vertex", std::to_string(selected_vertex).c_str())) {
            for (uint vertex : excitable_vertices) {
                if (Selectable(std::to_string(vertex).c_str(), vertex == selected_vertex)) {
                    action = SoundObjectAction::SelectVertex{vertex};
                }
            }
            EndCombo();
        }
        const bool can_excite = (impact_mode && ImpactModel->CanExcite()) || (modal_mode && ModalModel->CanExcite());
        if (!can_excite) BeginDisabled();
        Button("Strike");
        if (IsItemActivated()) action = SoundObjectAction::SetExciteForce{1.f};
        else if (IsItemDeactivated()) action = SoundObjectAction::SetExciteForce{0.f};
        if (!can_excite) EndDisabled();
    }
    if (impact_mode && ImpactModel) {
        ImpactModel->Draw();
    } else if (modal_mode) {
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
                    // Save wav files for both the modal and real-world impact sounds.
                    static const auto WavOutDir = fs::path{".."} / "audio_samples";
                    modal.WriteWav(WavOutDir / std::format("{}-modal", name));
                    impact.WriteWav(WavOutDir / std::format("{}-impact", name));
                }
            }
        }

        SeparatorText("Material properties");
        if (BeginCombo("Presets", material.Name.c_str())) {
            for (const auto &material_choice : materials::acoustic::All) {
                const bool is_selected = (material_choice.Name == material.Name);
                if (Selectable(material_choice.Name.c_str(), is_selected)) {
                    material = material_choice;
                }
                if (is_selected) SetItemDefaultFocus();
            }
            EndCombo();
        }

        auto &material_props = material.Properties;
        Text("Density (kg/m^3)");
        InputDouble("##Density", &material_props.Density, 0.0f, 0.0f, "%.3f");
        Text("Young's modulus (Pa)");
        InputDouble("##Young's modulus", &material_props.YoungModulus, 0.0f, 0.0f, "%.3f");
        Text("Poisson's ratio");
        InputDouble("##Poisson's ratio", &material_props.PoissonRatio, 0.0f, 0.0f, "%.3f");
        Text("Rayleigh damping alpha/beta");
        InputDouble("##Rayleigh damping alpha", &material_props.Alpha, 0.0f, 0.0f, "%.3f");
        InputDouble("##Rayleigh damping beta", &material_props.Beta, 0.0f, 0.0f, "%.3f");

        if (DspGenerator) {
            if (auto m2f_result = DspGenerator->Render()) {
                DspGenerator.reset();
                ModalModel = std::make_unique<ModalAudioModel>(std::move(*m2f_result), CreateSvg);
                action = SoundObjectAction::SetModel{SoundObjectModel::Modal};
            }
        }
        static int num_excitable_vertices = 10;
        if (num_excitable_vertices > tets->numberofpoints) num_excitable_vertices = tets->numberofpoints;
        SliderInt("Num excitable vertices", &num_excitable_vertices, 1, tets->numberofpoints);

        if (Button(std::format("{} DSP", ModalModel ? "Regenerate" : "Generate").c_str())) {
            // Linearly distribute the vertices across the tet mesh.
            DspGenerator = std::make_unique<Worker<Mesh2FaustResult>>("Generating DSP code...", [&] {
                auto excitable_vertices = iota_view{0u, uint(num_excitable_vertices)} | transform([&](uint i) { return i * tets->numberofpoints / num_excitable_vertices; }) | to<std::vector<uint>>();
                std::optional<float> fundamental_freq = ImpactModel && ImpactModel->Waveform ? std::optional{ImpactModel->Waveform->GetPeakFrequencies(10).front()} : std::nullopt;
                return GenerateDsp(*tets, material_props, std::move(excitable_vertices), true, fundamental_freq);
            });
        }
    }

    return action;
}
