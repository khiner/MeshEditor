#include "SoundObject.h"

#include <format>
#include <print>
#include <ranges>

#include "imgui.h"
#include "implot.h"

#include "mesh2faust.h"

using Sample = float;
#ifndef FAUSTFLOAT
#define FAUSTFLOAT Sample
#endif

#include "faust/dsp/llvm-dsp.h"

#include "FFTData.h"
#include "FaustParams.h"

#include "tetMesh.h" // Vega
#include "tetgen.h" // Must be after any Faust includes, since it defined a `REAL` macro.

#include "Tets.h"
#include "Worker.h"

using std::string, std::string_view;
using std::ranges::iota_view;
using std::ranges::to;
using std::views::transform;

void ApplyCosineWindow(float *w, uint n, const float *coeff, uint ncoeff) {
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
std::vector<float> CreateBlackmanHarris(uint n) {
    std::vector<float> window(n);
    static const float coeff[4] = {0.35875, -0.48829, 0.14128, -0.01168};
    ApplyCosineWindow(window.data(), n, coeff, sizeof(coeff) / sizeof(float));
    return window;
}

std::vector<float> ApplyWindow(const std::vector<float> &window, const float *data) {
    std::vector<float> windowed_data(window.size());
    for (uint i = 0; i < window.size(); ++i) windowed_data[i] = window[i] * data[i];
    return windowed_data;
}

constexpr uint SampleRate = 48000; // todo respect device sample rate

// Ordered by lowest to highest frequency.
static std::vector<float> FindPeakFrequencies(const fftwf_complex *data, uint n_bins, uint n_peaks) {
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

struct Waveform {
    // Capture a short audio segment shortly after the impact for FFT.
    static constexpr uint FftStartFrame = 30, FftEndFrame = SampleRate / 16;
    inline static auto BHWindow = CreateBlackmanHarris(FftEndFrame - FftStartFrame);
    const std::vector<float> Frames;
    std::vector<float> WindowedFrames;
    const FFTData FftData;
    std::vector<float> PeakFrequencies;

    Waveform(const float *frames, uint frame_count)
        : Frames(frames, frames + frame_count),
          WindowedFrames(ApplyWindow(BHWindow, Frames.data() + FftStartFrame)), FftData(WindowedFrames) {}

    void PlotFrames(const std::string &label = "Waveform") const;
    void PlotMagnitudeSpectrum(const std::string &label = "Magnitude spectrum", std::optional<uint> highlight_peak_freq_index = std::nullopt) const;

    void FindPeakFrequencies(uint n_peaks) {
        PeakFrequencies = ::FindPeakFrequencies(FftData.Complex, WindowedFrames.size(), n_peaks);
    }
};

// `FaustDSP` is a wrapper around a Faust DSP and Box.
// It has a Faust DSP code string, and updates its DSP and Box instances to reflect the current code.
struct FaustDSP {
    FaustDSP(string_view code) {
        SetCode(std::move(code));
    }
    ~FaustDSP() {
        Uninit();
    }

    inline static const string FaustDspFileExtension = ".dsp";

    Box Box{nullptr};
    dsp *Dsp{nullptr};
    std::unique_ptr<FaustParams> Params;

    string ErrorMessage{""};

    void SetCode(string_view code) {
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

private:
    string Code{""};
    llvm_dsp_factory *DspFactory{nullptr};

    void Init() {
        if (Code.empty()) return;

        createLibContext();

        static const string AppName = "MeshEditor";
        static const string LibrariesPath = fs::relative("../lib/faust/libraries");
        std::vector<const char *> argv = {"-I", LibrariesPath.c_str()};
        if (std::is_same_v<Sample, double>) argv.push_back("-double");
        const int argc = argv.size();

        static int num_inputs, num_outputs;
        Box = DSPToBoxes(AppName, Code, argc, argv.data(), &num_inputs, &num_outputs, ErrorMessage);

        if (Box && ErrorMessage.empty()) {
            static const int optimize_level = -1;
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

SoundObjectData::ImpactAudio::ImpactAudio(std::unordered_map<uint, std::vector<float>> &&impact_frames_by_vertex)
    : ImpactFramesByVertex(std::move(impact_frames_by_vertex)) {
    // Start at the end of the first sample, so it doesn't immediately play.
    // All samples are the same length.
    if (!ImpactFramesByVertex.empty()) CurrentFrame = ImpactFramesByVertex.begin()->second.size();
}
SoundObjectData::ImpactAudio::~ImpactAudio() = default;

const SoundObjectData::ImpactAudio &SoundObjectData::ImpactAudio::operator=(ImpactAudio &&other) noexcept {
    if (this != &other) {
        ImpactFramesByVertex = std::move(other.ImpactFramesByVertex);
        CurrentFrame = other.CurrentFrame;
        Waveform = std::move(other.Waveform);
    }
    return *this;
}

void SoundObjectData::ImpactAudio::SetVertex(uint vertex) {
    if (ImpactFramesByVertex.contains(vertex)) {
        auto &frames = ImpactFramesByVertex.at(vertex);
        Waveform = std::make_unique<::Waveform>(frames.data(), frames.size());
        Waveform->FindPeakFrequencies(10);
    }
}

SoundObjectData::Modal::Modal() {}
SoundObjectData::Modal::~Modal() = default;

void SoundObjectData::Modal::Set(const Mesh2FaustResult &m2f) {
    FaustDsp = std::make_unique<FaustDSP>(m2f.ModelDsp);
    ModeFreqs = std::move(m2f.ModeFreqs);
    ModeT60s = std::move(m2f.ModeT60s);
    ModeGains = std::move(m2f.ModeGains);
}

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

    static const string model_name = "modalModel";
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
            .modesMaxFreq = 20000,
            .targetNModes = 50, // number of synthesized modes
            .femNModes = 100, // number of modes to be computed for the finite element analysis
            // Convert to signed ints.
            .exPos = excitable_vertex_indices | transform([](uint i) { return int(i); }) | to<std::vector>(),
            .nExPos = int(excitable_vertex_indices.size()),
            .debugMode = false,
        }
    );
    const string model_dsp = m2f_result.modelDsp;
    if (model_dsp.empty()) return {"process = 0;", {}, {}, {}};

    auto &mode_freqs = m2f_result.model.modeFreqs;
    const float fundamental_freq = fundamental_freq_opt ?
        *fundamental_freq_opt :
        !mode_freqs.empty() ? mode_freqs.front() :
                              440.0f;

    // Static code sections.
    static const string to_sandh = " : ba.sAndH(gate);"; // Add a sample and hold on the gate, in serial, and end the expression.
    static const string
        gain = "gain = hslider(\"gain[scale:log]\",0.2,0,0.5,0.01);",
        t60_scale = "t60Scale = hslider(\"t60[scale:log][tooltip: Scale T60 decay values of all modes by the same amount.]\",1,0.1,10,0.01)" + to_sandh,
        gate = "gate = button(\"gate[tooltip: When excitation source is 'Hammer', excites the vertex. With any excitation source, applies the current parameters.]\");",
        hammer_hardness = "hammerHardness = hslider(\"hammerHardness[tooltip: Only has an effect when excitation source is 'Hammer'.]\",0.9,0,1,0.01)" + to_sandh,
        hammer_size = "hammerSize = hslider(\"hammerSize[tooltip: Only has an effect when excitation source is 'Hammer'.]\",0.3,0,1,0.01)" + to_sandh,
        hammer = "hammer(trig,hardness,size) = en.ar(att,att,trig)*no.noise : fi.lowpass(3,ctoff)\nwith{ ctoff = (1-size)*9500+500; att = (1-hardness)*0.01+0.001; };";

    // Variable code sections.
    const uint num_excite_pos = excitable_vertex_indices.size();
    const string
        freq = std::format("freq = hslider(\"Frequency[scale:log][tooltip: Fundamental frequency of the model]\",{},60,26000,1){}", fundamental_freq, to_sandh),
        ex_pos = std::format("exPos = nentry(\"exPos\",{},0,{},1){}", (num_excite_pos - 1) / 2, num_excite_pos - 1, to_sandh),
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
        .ModeGains = std::move(m2f_result.model.modeGains)
    };
}

MaterialProperties GetMaterialPreset(const std::string &name) {
    if (MaterialPresets.contains(name)) return MaterialPresets.at(name);
    return MaterialPresets.at(DefaultMaterialPresetName);
}

SoundObject::SoundObject(
    const ::Tets &tets, const std::optional<std::string> &material_name, vec3 listener_position, uint listener_entity_id
) : Tets(tets), MaterialName(material_name.value_or(DefaultMaterialPresetName)), Material(GetMaterialPreset(MaterialName)),
    ListenerPosition(std::move(listener_position)),
    ListenerEntityId(listener_entity_id), ModalModel(std::in_place) {}

SoundObject::~SoundObject() = default;

void SoundObject::SetImpactFrames(std::unordered_map<uint, std::vector<float>> &&impact_frames_by_vertex) {
    ExcitableVertices.clear();
    if (!impact_frames_by_vertex.empty()) {
        ImpactAudioModel = std::move(impact_frames_by_vertex);
        for (auto &[vertex, _] : ImpactAudioModel->ImpactFramesByVertex) ExcitableVertices.emplace_back(vertex);
        CurrentVertex = ExcitableVertices.front();
        ImpactAudioModel->SetVertex(CurrentVertex);
    }
}

void SoundObjectData::Modal::ProduceAudio(float *input, float *output, uint frame_count) const {
    if (FaustDsp) FaustDsp->Compute(frame_count, &input, &output);
}

void SoundObject::ProduceAudio(DeviceData device, float *input, float *output, uint frame_count) {
    if (Model == SoundObjectModel::ImpactAudio && ImpactAudioModel) {
        if (!ImpactAudioModel->ImpactFramesByVertex.contains(CurrentVertex)) return;

        const auto &impact_samples = ImpactAudioModel->ImpactFramesByVertex.at(CurrentVertex);
        const uint sample_rate = device.SampleRate; // todo - resample from 48kHz to device sample rate if necessary
        (void)sample_rate; // Unused

        for (uint i = 0; i < frame_count; ++i) {
            output[i] += ImpactAudioModel->CurrentFrame < impact_samples.size() ? impact_samples[ImpactAudioModel->CurrentFrame++] : 0.0f;
        }
    } else if (Model == SoundObjectModel::Modal && ModalModel) {
        auto *dsp = ModalModel->FaustDsp.get();
        if (dsp && ImpactRecording && ImpactRecording->CurrentFrame == 0) dsp->Set("gate", 1);
        ModalModel->ProduceAudio(input, output, frame_count);
        if (ImpactRecording && !ImpactRecording->Complete) {
            for (uint i = 0; i < frame_count && ImpactRecording->CurrentFrame < ImpactRecording::FrameCount; ++i, ++ImpactRecording->CurrentFrame) {
                ImpactRecording->Frames[ImpactRecording->CurrentFrame] = output[i];
            }
            if (ImpactRecording->CurrentFrame == ImpactRecording::FrameCount) {
                ImpactRecording->Complete = true;
                dsp->Set("gate", 0);
            }
        }
    }
}

void SoundObject::SetModel(SoundObjectModel model) {
    Model = model;
}

using namespace ImGui;

void SoundObjectData::Modal::Draw() const {
    if (!FaustDsp) return;

    SeparatorText("DSP");
    if (Button("Print DSP code")) std::println("DSP code:\n\n{}\n", FaustDsp->GetCode());
    FaustDsp->DrawParams();
}

static const ImVec2 ChartSize = {-1, 160};

static float LinearToDb(float linear) { return 20.0f * log10f(linear); }

void Waveform::PlotFrames(const std::string &label) const {
    if (ImPlot::BeginPlot(label.c_str(), ChartSize)) {
        ImPlot::SetupAxes("Frame", "Amplitude");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, Frames.size(), ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.1, 1.1, ImGuiCond_Always);
        ImPlot::PushStyleVar(ImPlotStyleVar_Marker, ImPlotMarker_None);
        ImPlot::PlotLine("", Frames.data(), Frames.size());
        ImPlot::PopStyleVar();
        ImPlot::EndPlot();
    }
}

void Waveform::PlotMagnitudeSpectrum(const std::string &label, std::optional<uint> highlight_peak_freq_index) const {
    if (ImPlot::BeginPlot(label.c_str(), ChartSize)) {
        static const float MIN_DB = -200;
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
        for (uint i = 0; i < PeakFrequencies.size(); ++i) {
            const bool is_highlighted = highlight_peak_freq_index && i == *highlight_peak_freq_index;
            const float freq = PeakFrequencies[i];
            if (is_highlighted) {
                ImPlot::PushStyleColor(ImPlotCol_Line, ImGui::GetStyleColorVec4(ImGuiCol_PlotLinesHovered));
                ImPlot::PlotInfLines("##Highlight", &freq, 1);
                ImPlot::PopStyleColor();
            } else {
                ImPlot::PlotInfLines("##Peak", &freq, 1);
            }
        }
        ImPlot::PlotShaded("", frequency.data(), magnitude.data(), N_2, MIN_DB);
        ImPlot::PopStyleColor();
        ImPlot::PopStyleVar();
        ImPlot::EndPlot();
    }
}

// Returns the index of the hovered mode, if any.
static std::optional<size_t> PlotModeData(
    const std::vector<float> &data, const std::string &label, const std::string &x_label, const std::string &y_label,
    std::optional<size_t> highlight_index = std::nullopt, std::optional<float> max_value_opt = std::nullopt
) {
    std::optional<size_t> hovered_index;
    if (ImPlot::BeginPlot(label.c_str(), ChartSize)) {
        static const double BarSize = 0.9;
        const float max_value = max_value_opt.value_or(*std::max_element(data.begin(), data.end()));
        ImPlot::SetupAxes(x_label.c_str(), y_label.c_str());
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

void SoundObject::RenderControls() {
    if (ImpactAudioModel) {
        PushID("AudioModel");
        int model = int(Model);
        bool model_changed = RadioButton("Recordings", &model, int(SoundObjectModel::ImpactAudio));
        SameLine();
        model_changed |= RadioButton("Modal", &model, int(SoundObjectModel::Modal));
        PopID();
        if (model_changed) SetModel(SoundObjectModel(model));
    }
    if (Model == SoundObjectModel::ImpactAudio && ImpactAudioModel) {
        if (Button("Strike")) ImpactAudioModel->CurrentFrame = 0;
        if (BeginCombo("Vertex", std::to_string(CurrentVertex).c_str())) {
            for (auto &[vertex, _] : ImpactAudioModel->ImpactFramesByVertex) {
                if (Selectable(std::to_string(vertex).c_str(), vertex == CurrentVertex)) {
                    CurrentVertex = vertex;
                    ImpactAudioModel->SetVertex(CurrentVertex);
                }
            }
            EndCombo();
        }
        if (ImpactAudioModel->Waveform) {
            ImpactAudioModel->Waveform->PlotFrames("Real-world impact waveform");
            ImpactAudioModel->Waveform->PlotMagnitudeSpectrum("Real-world impact spectrum");
        }
    } else if (Model == SoundObjectModel::Modal && ModalModel) {
        if (auto &dsp = ModalModel->FaustDsp) {
            const bool is_recording = ImpactRecording && !ImpactRecording->Complete;
            if (is_recording) BeginDisabled();
            Button("Strike");
            if (IsItemActivated() && dsp->Get("gate") == 0) dsp->Set("gate", 1);
            else if (IsItemDeactivated() && dsp->Get("gate") == 1) dsp->Set("gate", 0);

            SameLine();
            if (Button("Record strike")) ImpactRecording = std::make_unique<::ImpactRecording>();
            if (is_recording) EndDisabled();

            if (ImpactRecording && ImpactRecording->Complete) {
                ModalModel->Waveform = std::make_unique<Waveform>(ImpactRecording->Frames, ImpactRecording::FrameCount);
                ModalModel->Waveform->FindPeakFrequencies(ModalModel->ModeFreqs.size());
                ImpactRecording.reset();
            }
            if (ModalModel->Waveform) {
                ModalModel->Waveform->PlotFrames("Modal impact waveform");
                ModalModel->Waveform->PlotMagnitudeSpectrum("Modal impact spectrum", HoveredModeIndex ? std::optional{*HoveredModeIndex} : std::nullopt);
            }

            // Poll the Faust DSP UI to see if the current excitation vertex has changed.
            const auto vertex_index = uint(dsp->Get("exPos"));
            if (vertex_index < ExcitableVertices.size()) CurrentVertex = ExcitableVertices[vertex_index];

            if (CollapsingHeader("Modal data charts")) {
                std::optional<size_t> new_hovered_index;
                if (auto hovered = PlotModeData(ModalModel->ModeFreqs, "Mode frequencies", "", "Frequency (Hz)", HoveredModeIndex)) new_hovered_index = hovered;
                if (auto hovered = PlotModeData(ModalModel->ModeT60s, "Mode T60s", "", "T60 decay time (s)", HoveredModeIndex)) new_hovered_index = hovered;
                if (auto hovered = PlotModeData(ModalModel->ModeGains[vertex_index], "Mode gains", "Mode index", "Gain", HoveredModeIndex, 1.f)) new_hovered_index = hovered;
                HoveredModeIndex = new_hovered_index;
                if (HoveredModeIndex && *HoveredModeIndex < ModalModel->ModeFreqs.size()) {
                    const auto hovered_index = *HoveredModeIndex;
                    Text(
                        "Mode %lu: Freq %.2f Hz, T60 %.2f s, Gain %.2f dB", hovered_index,
                        ModalModel->ModeFreqs[hovered_index],
                        ModalModel->ModeT60s[hovered_index],
                        ModalModel->ModeGains[vertex_index][hovered_index]
                    );
                }
            }
        }

        if (ModalModel) ModalModel->Draw();

        SeparatorText("Material properties");
        if (BeginCombo("Presets", MaterialName.c_str())) {
            for (const auto &[preset_name, material] : MaterialPresets) {
                const bool is_selected = (preset_name == MaterialName);
                if (Selectable(preset_name.c_str(), is_selected)) {
                    MaterialName = preset_name;
                    Material = material;
                }
                if (is_selected) SetItemDefaultFocus();
            }
            EndCombo();
        }

        Text("Density (kg/m^3)");
        InputDouble("##Density", &Material.Density, 0.0f, 0.0f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);
        Text("Young's modulus (Pa)");
        InputDouble("##Young's modulus", &Material.YoungModulus, 0.0f, 0.0f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);
        Text("Poisson's ratio");
        InputDouble("##Poisson's ratio", &Material.PoissonRatio, 0.0f, 0.0f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);
        Text("Rayleigh damping alpha/beta");
        InputDouble("##Rayleigh damping alpha", &Material.Alpha, 0.0f, 0.0f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);
        InputDouble("##Rayleigh damping beta", &Material.Beta, 0.0f, 0.0f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);
        if (DspGenerator) {
            if (const auto m2f_result = DspGenerator->Render()) {
                ModalModel->Set(*m2f_result);
                DspGenerator.reset();
            }
        }
        if (ModalModel) {
            if (Button(std::format("{} DSP", ModalModel->FaustDsp ? "Regenerate" : "Generate").c_str())) {
                if (!ImpactAudioModel) {
                    // ImpactAudio objects can only be struck at the impact points.
                    // Otherwise, linearly distribute the vertices across the tet mesh.
                    const uint num_excitable_vertices = 5; // todo UI input
                    ExcitableVertices = iota_view{0u, num_excitable_vertices} | transform([&](uint i) { return i * Tets->numberofpoints / num_excitable_vertices; }) | to<std::vector>();
                }
                std::optional<float> fundamental_freq = ImpactAudioModel && ImpactAudioModel->Waveform ? std::optional{ImpactAudioModel->Waveform->PeakFrequencies.front()} : std::nullopt;
                DspGenerator = std::make_unique<Worker<Mesh2FaustResult>>("Generating DSP code...", [&] {
                    return GenerateDsp(*Tets, Material, ExcitableVertices, true, fundamental_freq);
                });
            }
        }
    }
}
