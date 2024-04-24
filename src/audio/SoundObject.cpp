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

struct Waveform {
    // Capture a short audio segment just after the impact for FFT.
    inline static const uint FftStartFrame = 20, FftEndFrame = SampleRate / 20;
    inline static auto BHWindow = CreateBlackmanHarris(FftEndFrame - FftStartFrame);
    const std::vector<float> Frames;
    std::vector<float> WindowedFrames;
    const FFTData FftData;

    Waveform(const float *frames, uint frame_count)
        : Frames(frames, frames + frame_count),
          WindowedFrames(ApplyWindow(BHWindow, Frames.data() + FftStartFrame)),
          FftData(WindowedFrames) {}
};

// `FaustDSP` is a wrapper around a Faust DSP and Box.
// It has a Faust DSP code string, and updates its DSP and Box instances to reflect the current code.
struct FaustDSP {
    FaustDSP() {
        Init();
    }
    ~FaustDSP() {
        Uninit();
    }

    inline static const string FaustDspFileExtension = ".dsp";

    Box Box{nullptr};
    dsp *Dsp{nullptr};
    std::unique_ptr<FaustParams> Ui;

    string ErrorMessage{""};

    void SetCode(string_view code) {
        Code = std::move(code);
        Update();
    }
    std::string GetCode() const { return Code; }

    void Compute(uint n, Sample **input, Sample **output) {
        if (Dsp != nullptr) Dsp->compute(n, input, output);
    }

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
                    Ui = std::make_unique<FaustParams>();
                    Dsp->buildUserInterface(Ui.get());
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
        Ui.reset();
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
    }
}

SoundObjectData::Modal::Modal() : FaustDsp(std::make_unique<FaustDSP>()) {}
SoundObjectData::Modal::~Modal() = default;

Mesh2FaustResult GenerateDsp(const tetgenio &tets, const MaterialProperties &material, const std::vector<uint> &excitable_vertex_indices, bool freq_control = false) {
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
    const float fundamental_freq = mode_freqs.empty() ? 440.0f : mode_freqs.front();

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
    ListenerEntityId(listener_entity_id), ModalData(std::in_place) {}

SoundObject::~SoundObject() = default;

void SoundObject::SetImpactFrames(std::unordered_map<uint, std::vector<float>> &&impact_frames_by_vertex) {
    ExcitableVertices.clear();
    if (!impact_frames_by_vertex.empty()) {
        ImpactAudioData = std::move(impact_frames_by_vertex);
        for (auto &[vertex, _] : ImpactAudioData->ImpactFramesByVertex) ExcitableVertices.emplace_back(vertex);
        CurrentVertex = ExcitableVertices.front();
        ImpactAudioData->SetVertex(CurrentVertex);
    }
}

void SoundObject::ProduceAudio(DeviceData device, float *input, float *output, uint frame_count) {
    if (Model == SoundObjectModel::ImpactAudio && ImpactAudioData) {
        if (!ImpactAudioData->ImpactFramesByVertex.contains(CurrentVertex)) return;

        const auto &impact_samples = ImpactAudioData->ImpactFramesByVertex.at(CurrentVertex);
        const uint sample_rate = device.SampleRate; // todo - resample from 48kHz to device sample rate if necessary
        (void)sample_rate; // Unused

        for (uint i = 0; i < frame_count; ++i) {
            output[i] += ImpactAudioData->CurrentFrame < impact_samples.size() ? impact_samples[ImpactAudioData->CurrentFrame++] : 0.0f;
        }
    } else if (ModalData && ModalData->FaustDsp) {
        if (ImpactRecording && ImpactRecording->CurrentFrame == 0) {
            auto &zone = *ModalData->FaustDsp->Ui->getZoneForLabel("gate");
            zone = 1;
        }
        ModalData->FaustDsp->Compute(frame_count, &input, &output);
        if (ImpactRecording && !ImpactRecording->Complete) {
            for (uint i = 0; i < frame_count && ImpactRecording->CurrentFrame < ImpactRecording::FrameCount; ++i, ++ImpactRecording->CurrentFrame) {
                ImpactRecording->Frames[ImpactRecording->CurrentFrame] = output[i];
            }
            if (ImpactRecording->CurrentFrame == ImpactRecording::FrameCount) {
                ImpactRecording->Complete = true;
                auto &zone = *ModalData->FaustDsp->Ui->getZoneForLabel("gate");
                zone = 0;
            }
        }
    }
}

void SoundObject::SetModel(SoundObjectModel model) {
    Model = model;
}

using namespace ImGui;

static const ImVec2 ChartSize = {-1, 160};

static void RenderWaveform(const Waveform &waveform, const std::string &label = "Waveform") {
    if (ImPlot::BeginPlot(label.c_str(), ChartSize)) {
        ImPlot::SetupAxes("Frame", "Amplitude");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, waveform.Frames.size(), ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.1, 1.1, ImGuiCond_Always);
        ImPlot::PushStyleVar(ImPlotStyleVar_Marker, ImPlotMarker_None);
        ImPlot::PlotLine("", waveform.Frames.data(), waveform.Frames.size());
        ImPlot::PopStyleVar();
        ImPlot::EndPlot();
    }
}

static float LinearToDb(float linear) { return 20.0f * log10f(linear); }

static void RenderMagnitudeSpectrum(const Waveform &waveform, const std::string &label = "Magnitude spectrum") {
    if (ImPlot::BeginPlot(label.c_str(), ChartSize)) {
        static const float MIN_DB = -200;
        const FFTData &fft = waveform.FftData;
        const uint N = waveform.WindowedFrames.size();
        const uint N_2 = N / 2;
        const float fs = SampleRate; // todo flexible sample rate
        const float fs_n = fs / float(N);

        static std::vector<float> frequency(N_2), magnitude(N_2);
        frequency.resize(N_2);
        magnitude.resize(N_2);

        const auto *data = fft.Complex;
        for (uint i = 0; i < N_2; i++) {
            frequency[i] = fs_n * float(i);
            const float mag_linear = sqrtf(data[i][0] * data[i][0] + data[i][1] * data[i][1]) / float(N_2);
            magnitude[i] = LinearToDb(mag_linear);
        }

        ImPlot::SetupAxes("Frequency (Hz)", "Magnitude (dB)");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, fs / 2, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, MIN_DB, 0, ImGuiCond_Always);
        ImPlot::PushStyleVar(ImPlotStyleVar_Marker, ImPlotMarker_None);
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImGui::GetStyleColorVec4(ImGuiCol_PlotHistogramHovered));
        ImPlot::PlotShaded("", frequency.data(), magnitude.data(), N_2, MIN_DB);
        ImPlot::PopStyleColor();
        ImPlot::PopStyleVar();
        ImPlot::EndPlot();
    }
}

void SoundObject::RenderControls() {
    if (ImpactAudioData) {
        PushID("AudioModel");
        int model = int(Model);
        bool model_changed = RadioButton("Recordings", &model, int(SoundObjectModel::ImpactAudio));
        SameLine();
        model_changed |= RadioButton("Modal", &model, int(SoundObjectModel::Modal));
        PopID();
        if (model_changed) SetModel(SoundObjectModel(model));
    }
    if (Model == SoundObjectModel::ImpactAudio && ImpactAudioData) {
        if (Button("Strike")) ImpactAudioData->CurrentFrame = 0;
        if (BeginCombo("Vertex", std::to_string(CurrentVertex).c_str())) {
            for (auto &[vertex, _] : ImpactAudioData->ImpactFramesByVertex) {
                if (Selectable(std::to_string(vertex).c_str(), vertex == CurrentVertex)) {
                    CurrentVertex = vertex;
                    ImpactAudioData->SetVertex(CurrentVertex);
                }
            }
            EndCombo();
        }
        if (ImpactAudioData->Waveform) {
            RenderWaveform(*ImpactAudioData->Waveform, "Real-world impact waveform");
            RenderMagnitudeSpectrum(*ImpactAudioData->Waveform, "Real-world impact spectrum");
        }
    } else if (ModalData) {
        if (ModalData->FaustDsp && ModalData->FaustDsp->Ui) {
            const bool is_recording = ImpactRecording && !ImpactRecording->Complete;
            if (is_recording) BeginDisabled();
            Button("Strike");
            auto &zone = *ModalData->FaustDsp->Ui->getZoneForLabel("gate");
            if (IsItemActivated() && zone == 0.0) zone = 1.0;
            else if (IsItemDeactivated() && zone == 1.0) zone = 0.0;

            SameLine();
            if (Button("Record strike")) ImpactRecording = std::make_unique<::ImpactRecording>();
            if (is_recording) EndDisabled();

            if (ImpactRecording && ImpactRecording->Complete) {
                ModalData->Waveform = std::make_unique<Waveform>(ImpactRecording->Frames, ImpactRecording::FrameCount);
                ImpactRecording.reset();
            }
            if (ModalData->Waveform) {
                RenderWaveform(*ModalData->Waveform, "Modal impact waveform");
                RenderMagnitudeSpectrum(*ModalData->Waveform, "Modal impact spectrum");
            }
        }

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

        if (ModalData->FaustDsp->Ui) {
            // Poll the Faust DSP UI to see if the current excitation vertex has changed.
            const auto ui_vertex = uint(*ModalData->FaustDsp->Ui->getZoneForLabel("exPos"));
            if (ui_vertex < ExcitableVertices.size()) CurrentVertex = ExcitableVertices[ui_vertex];

            if (CollapsingHeader("Modal data charts")) {
                if (ImPlot::BeginPlot("Mode frequencies", ChartSize)) {
                    const auto &mode_freqs = ModalData->ModeFreqs;
                    const float max_value = *std::max_element(mode_freqs.begin(), mode_freqs.end());
                    ImPlot::SetupAxes("Mode index", "Frequency (Hz)");
                    // ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
                    ImPlot::SetupAxesLimits(-0.5f, mode_freqs.size() - 0.5f, 0, max_value, ImPlotCond_Always);
                    ImPlot::PlotBars("", mode_freqs.data(), mode_freqs.size(), 0.9);
                    ImPlot::EndPlot();
                }
                if (ImPlot::BeginPlot("Mode T60s", ChartSize)) {
                    const auto &mode_t60s = ModalData->ModeT60s;
                    const float max_value = *std::max_element(mode_t60s.begin(), mode_t60s.end());
                    ImPlot::SetupAxes("Mode index", "T60 decay time (s)");
                    ImPlot::SetupAxesLimits(-0.5f, mode_t60s.size() - 0.5f, 0, max_value, ImPlotCond_Always);
                    ImPlot::PlotBars("", mode_t60s.data(), mode_t60s.size(), 0.9);
                    ImPlot::EndPlot();
                }
                if (ImPlot::BeginPlot("Mode gains", ChartSize)) {
                    const auto curr_vertex_it = std::ranges::find(ExcitableVertices, CurrentVertex);
                    const auto current_mode = std::distance(ExcitableVertices.begin(), curr_vertex_it);
                    const auto &mode_gains = ModalData->ModeGains[current_mode];
                    ImPlot::SetupAxes("Mode index", "Gain");
                    ImPlot::SetupAxesLimits(-0.5f, mode_gains.size() - 0.5f, 0, 1, ImPlotCond_Always);
                    ImPlot::PlotBars("", mode_gains.data(), mode_gains.size(), 0.9);
                    ImPlot::EndPlot();
                }
            }
            SeparatorText("DSP control");
            if (Button("Print DSP code")) std::println("DSP code:\n\n{}\n", ModalData->FaustDsp->GetCode());
            ModalData->FaustDsp->Ui->Draw();
        } else {
            if (DspGenerator) {
                if (auto m2f_result = DspGenerator->Render()) {
                    ModalData->FaustDsp->SetCode(m2f_result->ModelDsp);
                    ModalData->ModeFreqs = std::move(m2f_result->ModeFreqs);
                    ModalData->ModeT60s = std::move(m2f_result->ModeT60s);
                    ModalData->ModeGains = std::move(m2f_result->ModeGains);
                    DspGenerator.reset();
                }
            } else if (Button("Generate DSP")) {
                if (!ImpactAudioData) {
                    // ImpactAudio objects can only be struck at the impact points.
                    // Otherwise, linearly distribute the vertices across the tet mesh.
                    const uint num_excitable_vertices = 5; // todo UI input
                    ExcitableVertices = iota_view{0u, num_excitable_vertices} | transform([&](uint i) { return i * Tets->numberofpoints / num_excitable_vertices; }) | to<std::vector>();
                }
                DspGenerator = std::make_unique<Worker<Mesh2FaustResult>>("Generating DSP code...", [&] {
                    return GenerateDsp(*Tets, Material, ExcitableVertices, true);
                });
            }
        }
    }
}
