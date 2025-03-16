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
#include "tetgen.h" // Must be after any Faust includes, since it defined a `REAL` macro.
#include <entt/entity/registry.hpp>

#include <format>
#include <optional>
#include <print>
#include <ranges>

using std::ranges::find, std::ranges::iota_view, std::ranges::sort, std::ranges::to;
using std::views::transform, std::views::take;

namespace {
struct Recording {
    Recording(uint frame_count) : Frames(frame_count) {}

    std::vector<float> Frames;
    uint Frame{0};

    bool Complete() const { return Frame == Frames.size(); }
    void Record(float value) {
        if (!Complete()) Frames[Frame++] = value;
    }
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
constexpr ImVec2 ChartSize{-1, 160};

// Capture a short audio segment shortly after the impact for FFT.
FFTData ComputeFft(const std::vector<float> &frames) {
    static constexpr uint FftStartFrame = 30, FftEndFrame = SampleRate / 16;
    static const auto BHWindow = CreateBlackmanHarris(FftEndFrame - FftStartFrame);
    return {ApplyWindow(BHWindow, frames.data() + FftStartFrame)};
}

std::vector<float> GetPeakFrequencies(const FFTData &fft_data, uint n_peaks) { return FindPeakFrequencies(fft_data.Complex, fft_data.NumReal, n_peaks); }

// If `normalize_max` is set, normalize the data to this maximum value.
void WriteWav(const std::vector<float> &frames, fs::path file_path, std::optional<float> normalize_max = std::nullopt) {
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
            ImPlot::PushStyleColor(ImPlotCol_Line, ImGui::GetStyleColorVec4(ImGuiCol_PlotLinesHovered));
            ImPlot::PlotInfLines("##Highlight", &*highlight_frame, 1);
            ImPlot::PopStyleColor();
        }
        ImPlot::PushStyleVar(ImPlotStyleVar_Marker, ImPlotMarker_None);
        ImPlot::PlotLine("", frames.data(), frames.size());
        ImPlot::PopStyleVar();
        ImPlot::EndPlot();
    }
}

void PlotMagnitudeSpectrum(const std::vector<float> &frames, std::string_view label = "Magnitude spectrum", std::optional<float> highlight_freq = {}) {
    static const std::vector<float> *frames_ptr{&frames};
    static FFTData fft_data{ComputeFft(frames)};
    if (&frames != frames_ptr) {
        fft_data = ComputeFft(frames);
        frames_ptr = &frames;
    }
    if (ImPlot::BeginPlot(label.data(), ChartSize)) {
        static constexpr float MinDb = -200;
        const uint N = fft_data.NumReal, N2 = N / 2;
        const float fs = SampleRate; // todo flexible sample rate
        const float fs_n = SampleRate / float(N);
        static std::vector<float> frequency(N2), magnitude(N2);
        frequency.resize(N2);
        magnitude.resize(N2);

        const auto *data = fft_data.Complex;
        for (uint i = 0; i < N2; i++) {
            frequency[i] = fs_n * float(i);
            magnitude[i] = LinearToDb(sqrtf(data[i][0] * data[i][0] + data[i][1] * data[i][1]) / float(N2));
        }

        ImPlot::SetupAxes("Frequency (Hz)", "Magnitude (dB)");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, fs / 2, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, MinDb, 0, ImGuiCond_Always);
        ImPlot::PushStyleVar(ImPlotStyleVar_Marker, ImPlotMarker_None);
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImGui::GetStyleColorVec4(ImGuiCol_PlotHistogramHovered));
        if (highlight_freq) {
            ImPlot::PushStyleColor(ImPlotCol_Line, ImGui::GetStyleColorVec4(ImGuiCol_PlotLinesHovered));
            ImPlot::PlotInfLines("##Highlight", &(*highlight_freq), 1);
            ImPlot::PopStyleColor();
        }
        ImPlot::PlotShaded("", frequency.data(), magnitude.data(), N2, MinDb);
        ImPlot::PopStyleColor();
        ImPlot::PopStyleVar();
        ImPlot::EndPlot();
    }
}
} // namespace

struct SampleAudioModel {
    SampleAudioModel(std::vector<std::vector<float>> &&frames, std::vector<uint> vertices)
        : Frames(std::move(frames)), Excitable(vertices) {}
    ~SampleAudioModel() = default;

    std::vector<std::vector<float>> Frames;
    Excitable Excitable;
    uint Frame{uint(GetFrames().size())}; // Don'a immediately play

    const std::vector<float> &GetFrames() const { return Frames[Excitable.SelectedVertexIndex]; }
    bool Complete() const { return Frame == GetFrames().size(); }
    void Stop() { Frame = GetFrames().size(); }
};

struct ModalAudioModel {
    std::unique_ptr<Recording> Recording;
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

SoundObject::SoundObject(FaustDSP &dsp) : Dsp(dsp) {}
SoundObject::~SoundObject() = default;

void SoundObject::SetImpactFrames(std::vector<std::vector<float>> &&impact_frames, std::vector<uint> &&vertex_indices) {
    if (!impact_frames.empty()) {
        SampleModel = std::make_unique<SampleAudioModel>(std::move(impact_frames), std::move(vertex_indices));
        Model = SoundObjectModel::Samples;
    }
}
void SoundObject::SetImpactFrames(std::vector<std::vector<float>> &&impact_frames) {
    if (SampleModel) {
        SampleModel->Stop();
        SampleModel->Frames = std::move(impact_frames);
    }
}

void SoundObject::ProduceAudio(AudioBuffer &buffer, entt::registry &r, entt::entity entity) const {
    if (Model == SoundObjectModel::Samples && SampleModel) {
        if (SampleModel->Frames.empty()) return;
        const auto &impact_samples = SampleModel->Frames[r.get<Excitable>(entity).SelectedVertexIndex];
        // todo - resample from 48kHz to device sample rate if necessary
        for (uint i = 0; i < buffer.FrameCount; ++i) {
            buffer.Output[i] += SampleModel->Frame < impact_samples.size() ? impact_samples[SampleModel->Frame++] : 0.0f;
        }
    } else if (Model == SoundObjectModel::Modal && ModalModel) {
        const auto &recording = ModalModel->Recording;
        if (!recording) return;

        if (recording->Frame == 0) Dsp.Set(GateParamName, 1);
        if (!recording->Complete()) {
            for (uint i = 0; i < buffer.FrameCount && !recording->Complete(); ++i) {
                recording->Record(buffer.Output[i]);
            }
            if (recording->Complete()) Dsp.Set(GateParamName, 0);
        }
    }
}

void SoundObject::SetVertex(uint vertex) {
    // Update vertex in all present models.
    if (SampleModel) {
        SampleModel->Stop();
        SampleModel->Excitable.SelectedVertexIndex = vertex;
    }
    if (ModalModel) {
        Dsp.Set(GateParamName, 0);
        Dsp.Set(ExciteIndexParamName, vertex);
    }
}
void SoundObject::SetVertexForce(float force) {
    // Update vertex force in the active model.
    if (Model == SoundObjectModel::Samples && SampleModel) {
        if (force > 0) SampleModel->Frame = 0;
    } else if (Model == SoundObjectModel::Modal && ModalModel) {
        Dsp.Set(GateParamName, force);
    }
}

void SoundObject::Stop() {
    if (SampleModel) SampleModel->Stop();
    if (ModalModel) Dsp.Set(GateParamName, 0);
}

void SoundObject::SetModel(SoundObjectModel model, entt::registry &r, entt::entity entity) {
    if (Model == model) return;

    Stop();
    Model = model;
    const bool is_impact = Model == SoundObjectModel::Samples && SampleModel;
    const bool is_modal = Model == SoundObjectModel::Modal && ModalModel;
    if (!is_impact && !is_modal) return;

    auto excitable = is_impact ? SampleModel->Excitable : r.get<ModalSoundObject>(entity).Excitable;
    r.emplace_or_replace<Excitable>(entity, std::move(excitable));
}

void SoundObject::Draw(entt::registry &r, entt::entity entity) {
    if (auto &dsp_generator = DspGenerator) {
        if (auto modal_sound_object = dsp_generator->Render()) {
            dsp_generator.reset();
            ModalModel = std::make_unique<ModalAudioModel>();
            Dsp.Set(ExciteIndexParamName, modal_sound_object->Excitable.SelectedVertexIndex);
            r.emplace_or_replace<ModalSoundObject>(entity, std::move(*modal_sound_object));
            SetModel(SoundObjectModel::Modal, r, entity);
        }
    }

    using namespace ImGui;

    auto new_model = Model;
    if (Model == SoundObjectModel::None) {
        if (SampleModel) new_model = SoundObjectModel::Samples;
        else if (ModalModel) new_model = SoundObjectModel::Modal;
    } else if (Model == SoundObjectModel::Samples && !SampleModel) {
        new_model = SoundObjectModel::Modal;
    } else if (Model == SoundObjectModel::Modal && !ModalModel) {
        new_model = SoundObjectModel::Samples;
    }
    if (SampleModel && ModalModel) {
        PushID("SelectAudioModel");
        auto model = int(new_model);
        bool model_changed = RadioButton("Recordings", &model, int(SoundObjectModel::Samples));
        SameLine();
        model_changed |= RadioButton("Modal", &model, int(SoundObjectModel::Modal));
        PopID();
        if (model_changed) new_model = SoundObjectModel(model);
    }
    if (new_model != Model) SetModel(new_model, r, entity);

    // Cross-model excite section
    auto *excitable = r.try_get<Excitable>(entity);
    if (excitable) {
        if (BeginCombo("Vertex", std::to_string(excitable->SelectedVertex()).c_str())) {
            const auto selected_vi = excitable->SelectedVertexIndex;
            for (uint vi = 0; vi < excitable->ExcitableVertices.size(); ++vi) {
                const auto vertex = excitable->ExcitableVertices[vi];
                if (Selectable(std::to_string(vertex).c_str(), vi == selected_vi)) {
                    r.remove<ExcitedVertex>(entity);
                    excitable->SelectedVertexIndex = vi;
                    SetVertex(vi);
                }
            }
            EndCombo();
        }
        const bool can_excite =
            (Model == SoundObjectModel::Samples) ||
            (Model == SoundObjectModel::Modal && (!ModalModel->Recording || ModalModel->Recording->Complete()));
        if (!can_excite) BeginDisabled();
        Button("Excite");
        if (IsItemActivated()) r.emplace<ExcitedVertex>(entity, excitable->SelectedVertex(), 1.f);
        else if (IsItemDeactivated()) r.remove<ExcitedVertex>(entity);
        if (!can_excite) EndDisabled();
    }

    // Impact model
    if (Model == SoundObjectModel::Samples) {
        SeparatorText("Real-world impact model");
        const auto &frames = SampleModel->GetFrames();
        PlotFrames(frames, "Waveform", SampleModel->Frame);
        PlotMagnitudeSpectrum(frames, "Spectrum");
    }

    // Model model create/edit (show even in impact/none mode)
    SeparatorText("Modal model");

    auto *parent_window = GetCurrentWindow();
    auto *create_info = r.try_get<ModalModelCreateInfo>(entity);
    if (!create_info) {
        // Open create/edit
        const auto *obj = r.try_get<const ModalSoundObject>(entity);
        if (Button(std::format("{} modal model", obj ? "Edit" : "Create").c_str())) {
            ModalModelCreateInfo create_info{};
            if (obj) {
                create_info.NumExcitableVertices = obj->Excitable.ExcitableVertices.size();
            }
            if (const auto *material = r.try_get<const AcousticMaterial>(entity)) {
                create_info.Material = *material;
            }
            r.emplace<ModalModelCreateInfo>(entity, std::move(create_info));
        }
    } else if (BeginChild("CreateModalAudioModel", ImVec2{-FLT_MIN, 0.f}, ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY, ImGuiWindowFlags_MenuBar)) {
        // Create/edit
        if (BeginMenuBar()) { // just title
            Text("Create modal audio model");
            EndMenuBar();
        }
        auto &info = *create_info;
        SeparatorText("Material properties");
        if (BeginCombo("Presets", info.Material.Name.c_str())) {
            for (const auto &material_choice : materials::acoustic::All) {
                const bool is_selected = (material_choice.Name == info.Material.Name);
                if (Selectable(material_choice.Name.c_str(), is_selected)) {
                    info.Material = material_choice;
                }
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

        SeparatorText("Excitable vertices");
        // If impact model is present, default the modal model to be excitable at exactly the same points.
        if (SampleModel) Checkbox("Use RealImpact vertices", &info.UseImpactVertices);

        const auto &mesh = r.get<const Mesh>(entity);
        if (!SampleModel || !info.UseImpactVertices) {
            const uint num_vertices = mesh.GetVertexCount();
            info.NumExcitableVertices = std::min(info.NumExcitableVertices, num_vertices);
            const uint min_vertices = 1, max_vertices = num_vertices;
            SliderScalar("Num excitable vertices", ImGuiDataType_U32, &info.NumExcitableVertices, &min_vertices, &max_vertices);
        }

        if (Button("Create")) {
            Stop();
            r.emplace_or_replace<AcousticMaterial>(entity, info.Material);
            const auto scale = r.get<Scale>(entity).Value;

            DspGenerator = std::make_unique<Worker<ModalSoundObject>>(parent_window, "Generating modal audio model...", [&, entity, scale] {
                // todo Add an invisible tet mesh to the scene and support toggling between surface/volumetric tet mesh views.
                // scene.AddMesh(tets->CreateMesh(), {.Name = "Tet Mesh", R.get<Model>(selected_entity).Transform;, .Select = false, .Visible = false});

                // We rely on `PreserveSurface` behavior for excitable vertices;
                // Vertex indices on the surface mesh must match vertex indices on the tet mesh.
                // todo display tet mesh in UI and select vertices for debugging (just like other meshes but restrict to edge view)

                // Use impact model vertices or linearly distribute the vertices across the tet mesh.
                const auto num_vertices = mesh.GetVertexCount();
                const auto excitable_vertices = SampleModel && info.UseImpactVertices ?
                    SampleModel->Excitable.ExcitableVertices :
                    iota_view{0u, uint(info.NumExcitableVertices)} | transform([&](uint i) { return i * num_vertices / info.NumExcitableVertices; }) | to<std::vector<uint>>();

                while (!DspGenerator) {}
                DspGenerator->SetMessage("Generating tetrahedral mesh...");
                const auto tets = GenerateTets(mesh, scale, {.PreserveSurface = true, .Quality = info.QualityTets});

                DspGenerator->SetMessage("Generating modal model...");
                const auto fundamental = SampleModel ? std::optional{GetPeakFrequencies(ComputeFft(SampleModel->GetFrames()), 10).front()} : std::nullopt;
                // const std::optional<float> fundamental = {};
                r.remove<ModalModelCreateInfo>(entity);
                ModalSoundObject obj{
                    .Modes = m2f::mesh2modes(*tets, info.Material.Properties, excitable_vertices, fundamental),
                    .Excitable = {excitable_vertices},
                };
                if (fundamental) obj.FundamentalFreq = *fundamental;
                return obj;
            });
        }
        SameLine();
        if (Button("Cancel")) r.remove<ModalModelCreateInfo>(entity);
        EndChild();
    }

    if (Model != SoundObjectModel::Modal) return;

    // Modal
    const auto *modal_sound_object = r.try_get<const ModalSoundObject>(entity);
    if (!excitable || !modal_sound_object) return;

    static std::optional<size_t> hovered_mode_index;
    const auto &model = *modal_sound_object;
    const auto &modes = model.Modes;
    if (ModalModel->Recording && ModalModel->Recording->Complete()) {
        const auto &frames = ModalModel->Recording->Frames;
        PlotFrames(frames, "Modal impact waveform");
        const auto highlight_freq = hovered_mode_index ? std::optional{modes.Freqs[*hovered_mode_index]} : std::nullopt;
        PlotMagnitudeSpectrum(frames, "Modal impact spectrum", highlight_freq);
    }

    // Poll the Faust DSP UI to see if the current excitation vertex has changed.
    excitable->SelectedVertexIndex = uint(Dsp.Get(ExciteIndexParamName));
    if (CollapsingHeader("Modal data charts")) {
        std::optional<size_t> new_hovered_index;
        const auto scaled_mode_freqs = modes.Freqs | transform([&](float f) { return model.FundamentalFreq * f / modes.Freqs.front(); }) | to<std::vector>();
        if (auto hovered = PlotModeData(scaled_mode_freqs, "Mode frequencies", "", "Frequency (Hz)", hovered_mode_index)) new_hovered_index = hovered;
        if (auto hovered = PlotModeData(modes.T60s, "Mode T60s", "", "T60 decay time (s)", hovered_mode_index)) new_hovered_index = hovered;
        if (auto hovered = PlotModeData(modes.Gains[excitable->SelectedVertexIndex], "Mode gains", "Mode index", "Gain", hovered_mode_index, 1.f)) new_hovered_index = hovered;
        if (hovered_mode_index = new_hovered_index; hovered_mode_index && *hovered_mode_index < modes.Freqs.size()) {
            const auto index = *hovered_mode_index;
            Text(
                "Mode %lu: Freq %.2f Hz, T60 %.2f s, Gain %.2f dB", index,
                scaled_mode_freqs[index],
                modes.T60s[index],
                modes.Gains[excitable->SelectedVertexIndex][index]
            );
        }
    }

    if (CollapsingHeader("DSP parameters")) Dsp.DrawParams();
    static const fs::path FaustSvgDir{"MeshEditor-svg"};
    if (CollapsingHeader("DSP graph")) Dsp.DrawGraph(FaustSvgDir);
    if (Button("Print DSP code")) std::println("DSP code:\n\n{}\n", Dsp.GetCode());

    auto &recording = ModalModel->Recording;
    const bool is_recording = recording && !recording->Complete();
    if (is_recording) BeginDisabled();
    static constexpr uint RecordFrames = 208'592; // Same length as RealImpact recordings.
    if (Button("Record strike")) recording = std::make_unique<::Recording>(RecordFrames);
    if (is_recording) EndDisabled();

    if (SampleModel && recording && recording->Complete()) {
        // const auto &modal = *ModalModel->FftData, &impact = SampleModel->FftData;
        // uint ModeCount() const { return modes.Freqs.size(); }
        // const uint n_test_modes = std::min(ModalModel->ModeCount(), 10u);
        // Uncomment to cache `n_test_modes` peak frequencies for display in the spectrum plot.
        // RMSE is abyssmal in most cases...
        // const float rmse = RMSE(GetPeakFrequencies(modal, n_test_modes), GetPeakFrequencies(impact, n_test_modes));
        // Text("RMSE of top %d mode frequencies: %f", n_test_modes, rmse);
        SameLine();
        if (Button("Save wav files")) {
            const auto name = GetName(r, entity);
            // Save wav files for both the modal and real-world impact sounds.
            static const auto WavOutDir = fs::path{".."} / "audio_samples";
            WriteWav(recording->Frames, WavOutDir / std::format("{}-modal", name));
            WriteWav(SampleModel->GetFrames(), WavOutDir / std::format("{}-impact", name));
        }
    }
}
