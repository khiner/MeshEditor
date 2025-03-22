#include "AcousticScene.h"
#include "Excitable.h"
#include "FFTData.h"
#include "FaustDSP.h"
#include "FaustGenerator.h"
#include "RealImpact.h"
#include "Registry.h"
#include "Scale.h"
#include "Scene.h"
#include "Tets.h"
#include "Widgets.h" // imgui
#include "Worker.h"
#include "mesh/Mesh.h"
#include "mesh/Primitives.h"

#include "implot.h"
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

// If an entity has this component, it is being listened to by `Listener`.
struct SoundObjectListener {
    entt::entity Listener;
};
struct SoundObjectListenerPoint {
    uint Index; // Index in the root listener point's children.
};

struct SampleSoundObject {
    SampleSoundObject(std::vector<std::vector<float>> &&frames, std::vector<uint> vertices)
        : Frames(std::move(frames)), Excitable(vertices) {}
    ~SampleSoundObject() = default;

    std::vector<std::vector<float>> Frames;
    Excitable Excitable;
    uint Frame{uint(GetFrames().size())}; // Don'a immediately play

    const std::vector<float> &GetFrames() const { return Frames[Excitable.SelectedVertexIndex]; }
    bool Complete() const { return Frame == GetFrames().size(); }
    void Stop() { Frame = GetFrames().size(); }
};

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

AcousticScene::AcousticScene(entt::registry &r, CreateSvgResource create_svg)
    : R(r), CreateSvg(std::move(create_svg)), Dsp(std::make_unique<FaustDSP>(std::move(create_svg))),
      FaustGenerator(std::make_unique<::FaustGenerator>(r, [this](std::string_view code) { Dsp->SetCode(code); })) {
    R.on_construct<ExcitedVertex>().connect<&AcousticScene::OnCreateExcitedVertex>(*this);
    R.on_destroy<ExcitedVertex>().connect<&AcousticScene::OnDestroyExcitedVertex>(*this);
}
AcousticScene::~AcousticScene() = default;

void AcousticScene::OnCreateExcitedVertex(entt::registry &r, entt::entity entity) {
    if (r.all_of<SoundObjectModel>(entity)) {
        const auto &excited_vertex = r.get<const ExcitedVertex>(entity);
        const auto &excitable = r.get<const Excitable>(entity);
        if (auto vi = excitable.FindVertexIndex(excited_vertex.Vertex)) {
            R.patch<Excitable>(entity, [vi](auto &e) { e.SelectedVertexIndex = *vi; });
            SetVertex(entity, *vi);
            SetVertexForce(entity, excited_vertex.Force);
        }
    }
}
void AcousticScene::OnDestroyExcitedVertex(entt::registry &r, entt::entity entity) {
    if (r.all_of<SoundObjectModel>(entity)) {
        SetVertexForce(entity, 0.f);
    }
}

void AcousticScene::LoadRealImpact(const fs::path &directory, Scene &scene) {
    if (!fs::exists(directory)) throw std::runtime_error(std::format("RealImpact directory does not exist: {}", directory.string()));

    scene.ClearMeshes();
    const auto object_entity = scene.AddMesh(
        directory / "transformed.obj",
        {
            .Name = *RealImpact::FindObjectName(directory),
            // RealImpact meshes are oriented with Z up, but MeshEditor uses Y up.
            .Rotation = glm::angleAxis(-float(M_PI_2), vec3{1, 0, 0}) * glm::angleAxis(float(M_PI), vec3{0, 0, 1}),
        }
    );

    std::vector<uint> vertex_indices(RealImpact::NumImpactVertices);
    {
        auto impact_positions = RealImpact::LoadPositions(directory);
        // RealImpact npy file has vertex indices, but the indices may have changed due to deduplication,
        // so we don't even load them. Instead, we look up by position here.
        const auto &mesh = R.get<Mesh>(object_entity);
        for (uint i = 0; i < impact_positions.size(); ++i) {
            vertex_indices[i] = uint(mesh.FindNearestVertex(ToOpenMesh(impact_positions[i])).idx());
        }
    }

    const auto listener_entity = scene.AddMesh(
        Cylinder(0.5f * RealImpact::MicWidthMm / 1000.f, RealImpact::MicLengthMm / 1000.f),
        {
            .Name = std::format("RealImpact Listeners: {}", R.get<Name>(object_entity).Value),
            .Select = false,
            .Visible = false,
        }
    );
    for (const auto &listener_point : RealImpact::LoadListenerPoints(directory)) {
        static const auto rot_z = glm::angleAxis(float(M_PI_2), vec3{0, 0, 1}); // Cylinder is oriended with center along the Y axis.
        const auto listener_instance_entity = scene.AddInstance(
            listener_entity,
            {
                .Name = std::format("RealImpact Listener: {}", listener_point.Index),
                .Position = listener_point.GetPosition(scene.World.Up, true),
                .Rotation = glm::angleAxis(glm::radians(float(listener_point.AngleDeg)), scene.World.Up) * rot_z,
                .Select = false,
            }
        );
        R.emplace<SoundObjectListenerPoint>(listener_instance_entity, listener_point.Index);

        static constexpr uint CenterListenerIndex = 263; // This listener point is roughly centered.
        if (listener_point.Index == CenterListenerIndex) {
            R.emplace<SoundObjectListener>(object_entity, listener_instance_entity);

            static const auto FindMaterial = [](std::string_view name) -> std::optional<AcousticMaterial> {
                for (const auto &material : materials::acoustic::All) {
                    if (material.Name == name) return material;
                }
                return {};
            };
            auto material_name = RealImpact::FindMaterialName(R.get<Name>(object_entity).Value);
            const auto real_impact_material = material_name ? FindMaterial(*material_name) : std::nullopt;
            if (real_impact_material) R.emplace<AcousticMaterial>(object_entity, *real_impact_material);
            R.emplace<Frozen>(object_entity);
            R.emplace<Excitable>(object_entity, vertex_indices);
            SetImpactFrames(object_entity, to<std::vector>(RealImpact::LoadSamples(directory, listener_point.Index)), std::move(vertex_indices));
        }
    }
}

void AcousticScene::ProduceAudio(AudioBuffer buffer) const {
    Dsp->Compute(buffer.FrameCount, &buffer.Input, &buffer.Output);
    for (const auto e : R.view<SoundObjectModel>()) {
        ProduceAudio(e, buffer);
    }
}

using namespace ImGui;

void AcousticScene::RenderControls(Scene &scene) {
    static const float CharWidth = CalcTextSize("A").x;

    const auto selected_entity = scene.GetSelectedEntity();
    if (!R.storage<SoundObjectModel>().empty() && CollapsingHeader("Sound objects")) {
        if (MeshEditor::BeginTable("Sound objects", 3)) {
            TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, CharWidth * 10);
            TableSetupColumn("Name");
            TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, CharWidth * 20);
            TableHeadersRow();
            entt::entity entity_to_select = entt::null, entity_to_delete = entt::null;
            for (auto entity : R.view<SoundObjectModel>()) {
                const bool is_selected = entity == selected_entity;
                PushID(uint(entity));
                TableNextColumn();
                AlignTextToFramePadding();
                if (is_selected) TableSetBgColor(ImGuiTableBgTarget_RowBg0, GetColorU32(ImGuiCol_TextSelectedBg));
                TextUnformatted(IdString(entity).c_str());
                TableNextColumn();
                TextUnformatted(R.get<Name>(entity).Value.c_str());
                TableNextColumn();
                if (is_selected) BeginDisabled();
                if (Button("Select")) entity_to_select = entity;
                if (is_selected) EndDisabled();
                SameLine();
                if (Button("Delete")) entity_to_delete = entity;
                if (const auto *sound_listener = R.try_get<SoundObjectListener>(entity)) {
                    if (Button("Select listener point")) entity_to_select = sound_listener->Listener;
                }
                PopID();
            }
            if (entity_to_select != entt::null) scene.SelectEntity(entity_to_select);
            if (entity_to_delete != entt::null) scene.DestroyEntity(entity_to_delete);
            EndTable();
        }
    }
    if (!R.storage<SoundObjectListenerPoint>().empty() && CollapsingHeader("Listener points")) {
        if (MeshEditor::BeginTable("Listener points", 3)) {
            TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, CharWidth * 10);
            TableSetupColumn("Name");
            TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, CharWidth * 16);
            TableHeadersRow();
            entt::entity entity_to_select = entt::null, entity_to_delete = entt::null;
            for (auto entity : R.view<SoundObjectListenerPoint>()) {
                const bool is_selected = entity == selected_entity;
                PushID(uint(entity));
                TableNextColumn();
                AlignTextToFramePadding();
                if (is_selected) TableSetBgColor(ImGuiTableBgTarget_RowBg0, GetColorU32(ImGuiCol_TextSelectedBg));
                TextUnformatted(IdString(entity).c_str());
                TableNextColumn();
                TextUnformatted(R.get<Name>(entity).Value.c_str());
                TableNextColumn();
                if (is_selected) BeginDisabled();
                if (Button("Select")) entity_to_select = entity;
                if (is_selected) EndDisabled();
                SameLine();
                if (Button("Delete")) entity_to_delete = entity;
                PopID();
            }
            if (entity_to_select != entt::null) scene.SelectEntity(entity_to_select);
            if (entity_to_delete != entt::null) scene.DestroyEntity(entity_to_delete);
            EndTable();
        }
    }
    if (selected_entity == entt::null) {
        TextUnformatted("No selection");
        return;
    }

    // Display the selected sound object (which could be the object listened to if a listener is selected).
    const auto FindSelectedSoundEntity = [&]() -> entt::entity {
        if (R.all_of<SoundObjectModel>(selected_entity)) return selected_entity;
        if (R.storage<SoundObjectListener>().empty()) return entt::null;
        for (const auto &[entity, listener] : R.view<const SoundObjectListener>().each()) {
            if (listener.Listener == selected_entity) return entity;
        }
        if (R.all_of<SoundObjectListenerPoint>(selected_entity)) return *R.view<SoundObjectModel>().begin();
        return entt::null;
    };
    const auto sound_entity = FindSelectedSoundEntity();
    if (sound_entity == entt::null) {
        if (Button("Create sound object")) {
            R.emplace<Frozen>(selected_entity);
            R.emplace<SoundObjectModel>(selected_entity, SoundObjectModel::Modal);
            R.emplace<ModalModelCreateInfo>(selected_entity);
        }
        // todo Create a sample sound object
        //  - Load and assign a sample to the whole object, or assign to vertices.
        return;
    }

    if (sound_entity != selected_entity && Button("Select sound object")) {
        scene.SelectEntity(sound_entity);
    }

    const auto *listener = R.try_get<SoundObjectListener>(sound_entity);
    if (listener && listener->Listener != selected_entity) {
        if (Button("Select listener point")) {
            scene.SelectEntity(listener->Listener);
        }
    }

    if (const auto *listener_point = R.try_get<SoundObjectListenerPoint>(selected_entity);
        listener_point && (!listener || selected_entity != listener->Listener)) {
        if (Button("Set listener point")) {
            SetImpactFrames(sound_entity, to<std::vector>(RealImpact::LoadSamples(R.get<Path>(sound_entity).Value.parent_path(), listener_point->Index)));
            R.emplace_or_replace<SoundObjectListener>(sound_entity, selected_entity);
        }
    }

    SeparatorText(std::format("Selected sound object: {}", GetName(R, sound_entity)).c_str());
    Draw(sound_entity);
    Spacing();
    if (Button("Delete sound object")) {
        R.remove<Frozen, Excitable, Recording, SoundObjectModel, ModalSoundObject, SampleSoundObject, ModalModelCreateInfo, SoundObjectListener>(sound_entity);
    }
}

/***** Sound object *****/

namespace {
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

void AcousticScene::SetImpactFrames(entt::entity entity, std::vector<std::vector<float>> &&impact_frames, std::vector<uint> &&vertex_indices) {
    if (!impact_frames.empty()) {
        R.emplace_or_replace<SampleSoundObject>(entity, std::move(impact_frames), std::move(vertex_indices));
        R.emplace_or_replace<SoundObjectModel>(entity, SoundObjectModel::Samples);
    }
}
void AcousticScene::SetImpactFrames(entt::entity entity, std::vector<std::vector<float>> &&impact_frames) {
    if (auto *sample_object = R.try_get<SampleSoundObject>(entity)) {
        sample_object->Stop();
        sample_object->Frames = std::move(impact_frames);
    }
}

void AcousticScene::ProduceAudio(entt::entity entity, AudioBuffer &buffer) const {
    if (!R.all_of<SoundObjectModel>(entity)) return;

    const auto model = R.get<SoundObjectModel>(entity);
    auto *sample_object = R.try_get<SampleSoundObject>(entity);
    const auto *modal_object = R.try_get<ModalSoundObject>(entity);
    if (model == SoundObjectModel::Samples && sample_object) {
        if (sample_object->Frames.empty()) return;
        const auto &impact_samples = sample_object->Frames[R.get<Excitable>(entity).SelectedVertexIndex];
        // todo - resample from 48kHz to device sample rate if necessary
        for (uint i = 0; i < buffer.FrameCount; ++i) {
            buffer.Output[i] += sample_object->Frame < impact_samples.size() ? impact_samples[sample_object->Frame++] : 0.0f;
        }
    } else if (model == SoundObjectModel::Modal && modal_object) {
        if (auto *recording = R.try_get<Recording>(entity)) {
            if (recording->Frame == 0) Dsp->Set(GateParamName, 1);
            if (!recording->Complete()) {
                for (uint i = 0; i < buffer.FrameCount && !recording->Complete(); ++i) {
                    recording->Record(buffer.Output[i]);
                }
                if (recording->Complete()) Dsp->Set(GateParamName, 0);
            }
        }
    }
}

void AcousticScene::SetVertex(entt::entity entity, uint vertex) {
    Stop(entity);
    // Update vertex in all present models.
    if (auto *sample_object = R.try_get<SampleSoundObject>(entity)) {
        sample_object->Excitable.SelectedVertexIndex = vertex;
    }
    if (R.all_of<ModalSoundObject>(entity)) {
        Dsp->Set(ExciteIndexParamName, vertex);
    }
}
void AcousticScene::SetVertexForce(entt::entity entity, float force) {
    const auto model = R.get<SoundObjectModel>(entity);
    // Update vertex force in the active model.
    if (model == SoundObjectModel::Samples && force > 0) {
        if (auto *sample_object = R.try_get<SampleSoundObject>(entity)) {
            sample_object->Frame = 0;
        }
    } else if (model == SoundObjectModel::Modal && R.all_of<ModalSoundObject>(entity)) {
        Dsp->Set(GateParamName, force);
    }
}

void AcousticScene::Stop(entt::entity entity) {
    if (auto *sample_object = R.try_get<SampleSoundObject>(entity)) sample_object->Stop();
    if (R.all_of<ModalSoundObject>(entity)) Dsp->Set(GateParamName, 0);
}

void AcousticScene::SetModel(entt::entity entity, SoundObjectModel model) {
    Stop(entity);

    const auto *sample_object = R.try_get<const SampleSoundObject>(entity);
    const auto *modal_object = R.try_get<const ModalSoundObject>(entity);
    const bool is_sample = model == SoundObjectModel::Samples && sample_object;
    const bool is_modal = model == SoundObjectModel::Modal && modal_object;
    if (!is_sample && !is_modal) return;

    R.emplace_or_replace<SoundObjectModel>(entity, model);
    R.emplace_or_replace<Excitable>(entity, is_sample ? sample_object->Excitable : modal_object->Excitable);
}

void AcousticScene::Draw(entt::entity entity) {
    if (auto &dsp_generator = DspGenerator) {
        if (auto modal_sound_object = dsp_generator->Render()) {
            dsp_generator.reset();
            Dsp->Set(ExciteIndexParamName, modal_sound_object->Excitable.SelectedVertexIndex);
            R.emplace_or_replace<ModalSoundObject>(entity, std::move(*modal_sound_object));
            SetModel(entity, SoundObjectModel::Modal);
        }
    }

    using namespace ImGui;

    const auto *sample_object = R.try_get<SampleSoundObject>(entity);
    const auto *modal_object = R.try_get<ModalSoundObject>(entity);
    auto model = R.get<SoundObjectModel>(entity);
    if (sample_object && modal_object) {
        PushID("SelectAudioModel");
        auto edit_model = int(model);
        bool model_changed = RadioButton("Recordings", &edit_model, int(SoundObjectModel::Samples));
        SameLine();
        model_changed |= RadioButton("Modal", &edit_model, int(SoundObjectModel::Modal));
        PopID();
        if (model_changed) {
            model = SoundObjectModel(edit_model);
            SetModel(entity, model);
        }
    }

    // Cross-model excite section
    auto *recording = R.try_get<Recording>(entity);
    const auto *excitable = R.try_get<const Excitable>(entity);
    if (excitable) {
        if (BeginCombo("Vertex", std::to_string(excitable->SelectedVertex()).c_str())) {
            const auto selected_vi = excitable->SelectedVertexIndex;
            for (uint vi = 0; vi < excitable->ExcitableVertices.size(); ++vi) {
                const auto vertex = excitable->ExcitableVertices[vi];
                if (Selectable(std::to_string(vertex).c_str(), vi == selected_vi)) {
                    R.remove<ExcitedVertex>(entity);
                    R.patch<Excitable>(entity, [vi](auto &e) { e.SelectedVertexIndex = vi; });
                    SetVertex(entity, vi);
                }
            }
            EndCombo();
        }
        const bool can_excite =
            (model == SoundObjectModel::Samples) ||
            (model == SoundObjectModel::Modal && (!recording || recording->Complete()));
        if (!can_excite) BeginDisabled();
        Button("Excite");
        if (IsItemActivated()) R.emplace<ExcitedVertex>(entity, excitable->SelectedVertex(), 1.f);
        else if (IsItemDeactivated()) R.remove<ExcitedVertex>(entity);
        if (!can_excite) EndDisabled();
    }

    if (model == SoundObjectModel::Samples) {
        SeparatorText("Sound samples");
        const auto &frames = sample_object->GetFrames();
        PlotFrames(frames, "Waveform", sample_object->Frame);
        PlotMagnitudeSpectrum(frames, "Spectrum");
    }

    // Model model create/edit (show even in impact/none mode)
    SeparatorText("Modal model");

    auto *parent_window = GetCurrentWindow();
    auto *create_info = R.try_get<ModalModelCreateInfo>(entity);
    if (!create_info) {
        // Open create/edit
        if (Button(std::format("{} modal model", modal_object ? "Edit" : "Create").c_str())) {
            ModalModelCreateInfo create_info{};
            if (modal_object) {
                create_info.NumExcitableVertices = modal_object->Excitable.ExcitableVertices.size();
            }
            if (const auto *material = R.try_get<const AcousticMaterial>(entity)) {
                create_info.Material = *material;
            }
            R.emplace<ModalModelCreateInfo>(entity, std::move(create_info));
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
        // If a sample object is present, default the modal model to be excitable at the same points.
        if (sample_object) Checkbox("Use sample vertices", &info.UseSampleVertices);

        const auto &mesh = R.get<const Mesh>(entity);
        if (!sample_object || !info.UseSampleVertices) {
            const uint num_vertices = mesh.GetVertexCount();
            info.NumExcitableVertices = std::min(info.NumExcitableVertices, num_vertices);
            const uint min_vertices = 1, max_vertices = num_vertices;
            SliderScalar("Num excitable vertices", ImGuiDataType_U32, &info.NumExcitableVertices, &min_vertices, &max_vertices);
        }

        if (Button(modal_object ? "Update" : "Create")) {
            Stop(entity);
            R.emplace_or_replace<AcousticMaterial>(entity, info.Material);
            DspGenerator = std::make_unique<Worker<ModalSoundObject>>(
                parent_window, "Generating modal audio model...",
                [this, entity, info]() {
                    R.remove<ModalModelCreateInfo>(entity);
                    return CreateModalSoundObject(entity, info);
                }
            );
        }
        SameLine();
        if (Button("Cancel")) R.remove<ModalModelCreateInfo>(entity);
        EndChild();
    }

    if (model != SoundObjectModel::Modal) return;

    // Modal
    const auto *modal_sound_object = R.try_get<const ModalSoundObject>(entity);
    if (!excitable || !modal_sound_object) return;

    static std::optional<size_t> hovered_mode_index;
    const auto &modal = *modal_sound_object;
    const auto &modes = modal.Modes;
    if (recording && recording->Complete()) {
        const auto &frames = recording->Frames;
        PlotFrames(frames, "Modal impact waveform");
        const auto highlight_freq = hovered_mode_index ? std::optional{modes.Freqs[*hovered_mode_index]} : std::nullopt;
        PlotMagnitudeSpectrum(frames, "Modal impact spectrum", highlight_freq);
    }

    // Poll the Faust DSP UI to see if the current excitation vertex has changed.
    const auto excite_index = uint(Dsp->Get(ExciteIndexParamName));
    if (excitable->SelectedVertexIndex != excite_index) {
        R.patch<Excitable>(entity, [excite_index](auto &e) { e.SelectedVertexIndex = excite_index; });
    }
    if (CollapsingHeader("Modal data charts")) {
        std::optional<size_t> new_hovered_index;
        const auto scaled_mode_freqs = modes.Freqs | transform([&](float f) { return modal.FundamentalFreq * f / modes.Freqs.front(); }) | to<std::vector>();
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

    if (CollapsingHeader("DSP parameters")) Dsp->DrawParams();
    static const fs::path FaustSvgDir{"MeshEditor-svg"};
    if (CollapsingHeader("DSP graph")) Dsp->DrawGraph(FaustSvgDir);
    if (Button("Print DSP code")) std::println("DSP code:\n\n{}\n", Dsp->GetCode());

    const bool is_recording = recording && !recording->Complete();
    if (is_recording) BeginDisabled();
    static constexpr uint RecordFrames = 208'592; // Same length as RealImpact recordings.
    if (Button("Record strike")) recording = &R.emplace<Recording>(entity, RecordFrames);
    if (is_recording) EndDisabled();

    if (sample_object && recording && recording->Complete()) {
        // const auto &modal = *modal_object->FftData, &impact = sample_object->FftData;
        // uint ModeCount() const { return modes.Freqs.size(); }
        // const uint n_test_modes = std::min(modal_object->ModeCount(), 10u);
        // Uncomment to cache `n_test_modes` peak frequencies for display in the spectrum plot.
        // RMSE is abyssmal in most cases...
        // const float rmse = RMSE(GetPeakFrequencies(modal, n_test_modes), GetPeakFrequencies(impact, n_test_modes));
        // Text("RMSE of top %d mode frequencies: %f", n_test_modes, rmse);
        SameLine();
        if (Button("Save wav files")) {
            const auto name = GetName(R, entity);
            // Save wav files for both the modal and real-world impact sounds.
            static const auto WavOutDir = fs::path{".."} / "audio_samples";
            WriteWav(recording->Frames, WavOutDir / std::format("{}-modal", name));
            WriteWav(sample_object->GetFrames(), WavOutDir / std::format("{}-impact", name));
        }
    }
}

ModalSoundObject AcousticScene::CreateModalSoundObject(entt::entity entity, const ModalModelCreateInfo &info) const {
    // todo Add an invisible tet mesh to the scene and support toggling between surface/volumetric tet mesh views.
    // scene.AddMesh(tets->CreateMesh(), {.Name = "Tet Mesh", R.get<Model>(selected_entity).Transform;, .Select = false, .Visible = false});

    // We rely on `PreserveSurface` behavior for excitable vertices;
    // Vertex indices on the surface mesh must match vertex indices on the tet mesh.
    // todo display tet mesh in UI and select vertices for debugging (just like other meshes but restrict to edge view)

    const auto &mesh = R.get<const Mesh>(entity);
    const auto *sample_object = R.try_get<const SampleSoundObject>(entity);
    const auto scale = R.get<Scale>(entity).Value;
    // Use impact model vertices or linearly distribute the vertices across the tet mesh.
    const auto num_vertices = mesh.GetVertexCount();
    const auto excitable_vertices = sample_object && info.UseSampleVertices ?
        sample_object->Excitable.ExcitableVertices :
        iota_view{0u, uint(info.NumExcitableVertices)} | transform([&](uint i) { return i * num_vertices / info.NumExcitableVertices; }) | to<std::vector<uint>>();

    while (!DspGenerator) {}
    DspGenerator->SetMessage("Generating tetrahedral mesh...");
    const auto tets = GenerateTets(mesh, scale, {.PreserveSurface = true, .Quality = info.QualityTets});

    DspGenerator->SetMessage("Generating modal model...");
    const auto fundamental = sample_object ? std::optional{GetPeakFrequencies(ComputeFft(sample_object->GetFrames()), 10).front()} : std::nullopt;
    ModalSoundObject obj{
        .Modes = m2f::mesh2modes(*tets, info.Material.Properties, excitable_vertices, fundamental),
        .Excitable = {excitable_vertices, 0},
    };
    if (fundamental) obj.FundamentalFreq = *fundamental;
    return obj;
}
