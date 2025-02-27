#include "SoundObject.h"

#include "AudioBuffer.h"
#include "Excitable.h"
#include "FFTData.h"
#include "FaustDSP.h"
#include "ModalSoundObject.h"
#include "Registry.h"
#include "Scale.h"
#include "Tets.h"
#include "Widgets.h" // imgui
#include "Worker.h"

#include "implot.h"
#include "mesh/Mesh.h"
#include "mesh2faust.h"
#include "miniaudio.h"
#include "tetMesh.h" // Vega
#include "tetgen.h" // Must be after any Faust includes, since it defined a `REAL` macro.
#include <entt/entity/registry.hpp>

#include <format>
#include <optional>
#include <print>
#include <ranges>

using std::ranges::find, std::ranges::iota_view, std::ranges::sort, std::ranges::to;
using std::views::transform, std::views::take;

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
    const uint N2 = n_bins / 2;

    std::vector<std::pair<float, uint>> peaks; // (magnitude, bin)
    for (uint i = 1; i < N2 - 1; i++) {
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
            static constexpr float MinDb = -200;
            const FFTData &fft = FftData;
            const uint N = WindowedFrames.size();
            const uint N2 = N / 2;
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
            ImPlot::PlotShaded("", frequency.data(), magnitude.data(), N2, MinDb);
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
    ImpactAudioModel(std::vector<std::vector<float>> &&impact_frames)
        : ImpactFrames(std::move(impact_frames)),
          // All samples are the same length.
          MaxFrame(ImpactFrames.empty() ? 0 : ImpactFrames.front().size()) {
        SetVertex(0);
    }
    ~ImpactAudioModel() = default;

    const ImpactAudioModel &operator=(ImpactAudioModel &&other) noexcept {
        if (this != &other) {
            ImpactFrames = std::move(other.ImpactFrames);
            Frame = other.Frame;
            Waveform = std::move(other.Waveform);
        }
        return *this;
    }

    std::vector<std::vector<float>> ImpactFrames;
    uint CurrentVertex{0};
    uint MaxFrame;
    uint Frame{MaxFrame}; // Start at the end, so it doesn't immediately play.
    std::unique_ptr<Waveform> Waveform; // Selected vertex's waveform

    void ProduceAudio(const AudioBuffer &buffer, uint selected_vertex_index) {
        if (ImpactFrames.empty()) return;

        const auto &impact_samples = ImpactFrames[selected_vertex_index];
        // todo - resample from 48kHz to device sample rate if necessary
        for (uint i = 0; i < buffer.FrameCount; ++i) {
            buffer.Output[i] += Frame < impact_samples.size() ? impact_samples[Frame++] : 0.0f;
        }
    }

    void Start() { Frame = 0; }
    void Stop() { Frame = MaxFrame; }
    bool IsStarted() const { return Frame != MaxFrame; }

    bool CanExcite() const { return bool(Waveform); }
    void SetVertex(uint vertex_index) {
        CurrentVertex = vertex_index;
        Stop();
        if (vertex_index < ImpactFrames.size()) {
            const auto &frames = ImpactFrames[CurrentVertex];
            Waveform = std::make_unique<::Waveform>(frames.data(), frames.size());
        }
    }
    void SetImpactFrames(std::vector<std::vector<float>> &&impact_frames) {
        if (ImpactFrames.size() != impact_frames.size()) return;

        ImpactFrames = std::move(impact_frames);
        SetVertex(CurrentVertex);
    }

    void SetVertexForce(float force) {
        if (force > 0 && !IsStarted()) Start();
        else if (force == 0 && IsStarted()) Stop();
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
std::unique_ptr<Worker<ModalSoundObject>> DspGenerator;
} // namespace

struct ModalAudioModel {
    ModalAudioModel(FaustDSP &dsp) : Dsp(dsp) {}
    ~ModalAudioModel() = default;

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
    void SetVertex(uint vertex_index) {
        Stop();
        SetParam(ExciteIndexParamName, vertex_index);
    }
    void SetVertexForce(float force) { SetParam(GateParamName, force); }
    void Stop() { SetVertexForce(0); }

    void SetParam(std::string_view param_label, Sample param_value) const {
        Dsp.Set(std::move(param_label), param_value);
    }

    void Draw(const ModalSoundObject &model, Excitable &excitable) {
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
        excitable.SelectedVertexIndex = uint(Dsp.Get(ExciteIndexParamName));
        if (CollapsingHeader("Modal data charts")) {
            std::optional<size_t> new_hovered_index;
            if (auto hovered = PlotModeData(model.ModeFreqs, "Mode frequencies", "", "Frequency (Hz)", HoveredModeIndex)) new_hovered_index = hovered;
            if (auto hovered = PlotModeData(model.ModeT60s, "Mode T60s", "", "T60 decay time (s)", HoveredModeIndex)) new_hovered_index = hovered;
            if (auto hovered = PlotModeData(model.ModeGains[excitable.SelectedVertexIndex], "Mode gains", "Mode index", "Gain", HoveredModeIndex, 1.f)) new_hovered_index = hovered;
            if (HoveredModeIndex = new_hovered_index; HoveredModeIndex && *HoveredModeIndex < model.ModeFreqs.size()) {
                const auto index = *HoveredModeIndex;
                Text(
                    "Mode %lu: Freq %.2f Hz, T60 %.2f s, Gain %.2f dB", index,
                    model.ModeFreqs[index],
                    model.ModeT60s[index],
                    model.ModeGains[excitable.SelectedVertexIndex][index]
                );
            }
        }

        if (CollapsingHeader("DSP parameters")) Dsp.DrawParams();
        static const fs::path FaustSvgDir{"MeshEditor-svg"};
        if (CollapsingHeader("DSP graph")) Dsp.DrawGraph(FaustSvgDir);
        if (Button("Print DSP code")) std::println("DSP code:\n\n{}\n", Dsp.GetCode());
    }

    std::unique_ptr<Waveform> Waveform{}; // Recorded waveform

    FaustDSP &Dsp;

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

SoundObject::SoundObject(AcousticMaterial material, FaustDSP &dsp) : Dsp(dsp) {
    Controls.Material = std::move(material);
}
SoundObject::~SoundObject() = default;

void SoundObject::SetImpactFrames(std::vector<std::vector<float>> &&impact_frames, std::vector<uint> &&vertex_indices) {
    if (!impact_frames.empty()) {
        ImpactVertices = std::move(vertex_indices);
        ImpactModel = std::make_unique<ImpactAudioModel>(std::move(impact_frames));
    }
}
void SoundObject::SetImpactFrames(std::vector<std::vector<float>> &&impact_frames) {
    if (ImpactModel) ImpactModel->SetImpactFrames(std::move(impact_frames));
}

void SoundObject::ProduceAudio(AudioBuffer &buffer, entt::registry &r, entt::entity entity) const {
    if (Model == SoundObjectModel::ImpactAudio && ImpactModel) {
        ImpactModel->ProduceAudio(buffer, r.get<Excitable>(entity).SelectedVertexIndex);
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
    if (Model == model) return;

    if (ImpactModel) ImpactModel->Stop();
    if (ModalModel) ModalModel->Stop();
    Model = model;
    const bool is_impact = Model == SoundObjectModel::ImpactAudio && ImpactModel && ImpactVertices;
    const bool is_modal = Model == SoundObjectModel::Modal && ModalModel;
    if (!is_impact && !is_modal) return;

    auto excitable = is_impact ? Excitable{*ImpactVertices, ImpactModel->CurrentVertex} : r.get<ModalSoundObject>(entity).Excitable;
    Controls.NumExcitableVertices = excitable.ExcitableVertices.size();
    r.emplace_or_replace<Excitable>(entity, std::move(excitable));
}

m2f::ModalModel GenerateModalModel(const tetgenio &tets, const AcousticMaterialProperties &material, const std::vector<uint> &excitable_vertices) {
    // Convert the tetrahedral mesh into a VegaFEM TetMesh.
    std::vector<int> tet_indices;
    tet_indices.reserve(tets.numberoftetrahedra * 4 * 3); // 4 triangles per tetrahedron, 3 indices per triangle.
    // Turn each tetrahedron into 4 triangles.
    for (uint i = 0; i < uint(tets.numberoftetrahedra); ++i) {
        auto &result_indices = tets.tetrahedronlist;
        uint tri_i = i * 4;
        int a = result_indices[tri_i], b = result_indices[tri_i + 1], c = result_indices[tri_i + 2], d = result_indices[tri_i + 3];
        tet_indices.insert(tet_indices.end(), {a, b, c, d, a, b, c, d, a, b, c, d});
    }
    TetMesh volumetric_mesh{
        tets.numberofpoints, tets.pointlist, tets.numberoftetrahedra * 3, tet_indices.data(),
        material.YoungModulus, material.PoissonRatio, material.Density
    };

    return m2f::mesh2modal(
        &volumetric_mesh,
        m2f::MaterialProperties{
            .youngModulus = material.YoungModulus,
            .poissonRatio = material.PoissonRatio,
            .density = material.Density,
            .alpha = material.Alpha,
            .beta = material.Beta
        },
        m2f::CommonArguments{
            .modelName = "", // Not used until code gen
            .freqControl = false, // Not used until code gen
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
}

void SoundObject::RenderControls(entt::registry &r, entt::entity entity) {
    if (auto &dsp_generator = DspGenerator) {
        if (auto modal_sound_object = dsp_generator->Render()) {
            dsp_generator.reset();
            ModalModel = std::make_unique<ModalAudioModel>(Dsp);
            ModalModel->SetVertex(modal_sound_object->Excitable.SelectedVertexIndex);
            r.emplace_or_replace<ModalSoundObject>(entity, std::move(*modal_sound_object));
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

    auto &excitable = r.get<Excitable>(entity);

    const bool impact_mode = Model == SoundObjectModel::ImpactAudio, modal_mode = Model == SoundObjectModel::Modal;
    if ((impact_mode && ImpactModel) || (modal_mode && ModalModel)) {
        const auto selected_vi = excitable.SelectedVertexIndex;
        if (BeginCombo("Vertex", std::to_string(excitable.SelectedVertex()).c_str())) {
            for (uint vi = 0; vi < excitable.ExcitableVertices.size(); ++vi) {
                const auto vertex = excitable.ExcitableVertices[vi];
                if (Selectable(std::to_string(vertex).c_str(), vi == selected_vi)) {
                    r.remove<ExcitedVertex>(entity);
                    excitable.SelectedVertexIndex = vi;
                    SetVertex(vi);
                }
            }
            EndCombo();
        }
        const bool can_excite = (impact_mode && ImpactModel->CanExcite()) || (modal_mode && ModalModel->CanExcite());
        if (!can_excite) BeginDisabled();
        Button("Strike");
        if (IsItemActivated()) {
            r.emplace<ExcitedVertex>(entity, excitable.SelectedVertex(), 1.f);
        } else if (IsItemDeactivated()) {
            r.remove<ExcitedVertex>(entity);
        }
        if (!can_excite) EndDisabled();
    }

    if (impact_mode) {
        if (ImpactModel && ImpactModel->Waveform) {
            ImpactModel->Waveform->PlotFrames("Real-world impact waveform", ImpactModel->Frame);
            ImpactModel->Waveform->PlotMagnitudeSpectrum("Real-world impact spectrum");
        }
        return;
    }

    const auto *modal_model = r.try_get<ModalSoundObject>(entity);
    // Modal mode
    if (ModalModel && modal_model) {
        ModalModel->Draw(*modal_model, excitable);
        if (ModalModel->Waveform && ImpactModel && ImpactModel->Waveform) {
            const auto &modal = *ModalModel->Waveform, &impact = *ImpactModel->Waveform;
            // uint ModeCount() const { return ModalModel.modeFreqs.size(); }
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
    if (BeginCombo("Presets", Controls.Material.Name.c_str())) {
        for (const auto &material_choice : materials::acoustic::All) {
            const bool is_selected = (material_choice.Name == Controls.Material.Name);
            if (Selectable(material_choice.Name.c_str(), is_selected)) {
                Controls.Material = material_choice;
            }
            if (is_selected) SetItemDefaultFocus();
        }
        EndCombo();
    }

    Text("Density (kg/m^3)");
    InputDouble("##Density", &Controls.Material.Properties.Density, 0.0f, 0.0f, "%.3f");
    Text("Young's modulus (Pa)");
    InputDouble("##Young's modulus", &Controls.Material.Properties.YoungModulus, 0.0f, 0.0f, "%.3f");
    Text("Poisson's ratio");
    InputDouble("##Poisson's ratio", &Controls.Material.Properties.PoissonRatio, 0.0f, 0.0f, "%.3f");
    Text("Rayleigh damping alpha/beta");
    InputDouble("##Rayleigh damping alpha", &Controls.Material.Properties.Alpha, 0.0f, 0.0f, "%.3f");
    InputDouble("##Rayleigh damping beta", &Controls.Material.Properties.Beta, 0.0f, 0.0f, "%.3f");

    const auto &original_material = r.get<AcousticMaterial>(entity);
    bool material_changed = Controls.Material.Properties != original_material.Properties;
    if (material_changed && Button("Reset material props")) {
        Controls.Material = original_material;
        material_changed = false;
    }

    SeparatorText("Tet mesh");

    Checkbox("Quality", &Controls.QualityTets);
    MeshEditor::HelpMarker("Add new Steiner points to the interior of the tet mesh to improve model quality.");

    SeparatorText("Excitable vertices");
    // If impact model is present, default the modal model to be excitable at exactly the same points.
    if (ImpactModel) Checkbox("Use RealImpact vertices", &Controls.UseImpactVertices);

    const auto &mesh = r.get<const Mesh>(entity);
    if (!ImpactModel || !Controls.UseImpactVertices) {
        const uint num_points = mesh.GetVertexCount();
        Controls.NumExcitableVertices = std::min(Controls.NumExcitableVertices, num_points);
        const uint MinExcitableVertices = 1, MaxExcitableVertices = num_points;
        SliderScalar("Num excitable vertices", ImGuiDataType_U32, &Controls.NumExcitableVertices, &MinExcitableVertices, &MaxExcitableVertices);
    }

    const bool disable_generate = !material_changed && modal_model && Controls.NumExcitableVertices == modal_model->Excitable.ExcitableVertices.size();
    if (disable_generate) BeginDisabled();
    if (Button(std::format("{} audio model", ModalModel ? "Regenerate" : "Generate").c_str())) {
        if (material_changed) r.replace<AcousticMaterial>(entity, Controls.Material);
        const auto scale = r.get<Scale>(entity).Value;
        DspGenerator = std::make_unique<Worker<ModalSoundObject>>("Generating modal audio model...", [&, scale] {
            // todo Add an invisible tet mesh to the scene and support toggling between surface/volumetric tet mesh views.
            // scene.AddMesh(tets->CreateMesh(), {.Name = "Tet Mesh", R.get<Model>(selected_entity).Transform;, .Select = false, .Visible = false});

            // We rely on `PreserveSurface` behavior for excitable vertices;
            // Vertex indices on the surface mesh must match vertex indices on the tet mesh.
            // todo display tet mesh in UI and select vertices for debugging (just like other meshes but restrict to edge view)

            const auto fundamental_freq = ImpactModel && ImpactModel->Waveform ? std::optional{ImpactModel->Waveform->GetPeakFrequencies(8).front()} : std::nullopt;
            // Use impact model vertices or linearly distribute the vertices across the tet mesh.
            const auto num_points = mesh.GetVertexCount();
            const auto excitable_vertices = ImpactModel && ImpactVertices && Controls.UseImpactVertices ?
                *ImpactVertices :
                iota_view{0u, uint(Controls.NumExcitableVertices)} | transform([&](uint i) { return i * num_points / Controls.NumExcitableVertices; }) | to<std::vector<uint>>();

            while (!DspGenerator) {}
            DspGenerator->SetMessage("Generating tetrahedral mesh...");
            const auto tets = GenerateTets(mesh, scale, {.PreserveSurface = true, .Quality = Controls.QualityTets});

            DspGenerator->SetMessage("Generating modal model...");
            auto modal_model = GenerateModalModel(*tets, Controls.Material.Properties, excitable_vertices);

            return ModalSoundObject{
                .ModeFreqs = std::move(modal_model.modeFreqs),
                .ModeT60s = std::move(modal_model.modeT60s),
                .ModeGains = std::move(modal_model.modeGains),
                .Excitable = {excitable_vertices},
                .FundamentalFreq = fundamental_freq,
            };
        });
    }
    if (disable_generate) EndDisabled();
}
