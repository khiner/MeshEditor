#include "audio/AudioUi.h"

#include "FileDialog.h"
#include "action/Audio.h"
#include "audio/AudioDevice.h"
#include "audio/AudioQueries.h"
#include "audio/FftAnalysis.h"
#include "audio/SoundVertices.h"
#include "audio/WavWriter.h"
#include "editor/AudioExcitation.h"
#include "editor/AudioJobs.h"
#include "implot.h"
#include "imspinner.h"
#include "mesh/MeshStore.h"
#include "render/Instance.h"
#include "selection/SelectionComponents.h"
#include "ui/ChoiceCombo.h"
#include "ui/FieldEdit.h"
#include "viewport/InteractionComponents.h"

#include <format>

using numeric::dvec3;

namespace fs = std::filesystem;
using SurfaceSolveConfig = fastfem::SurfaceSolveConfig;
using ModalSolverConfig = fastfem::SolverConfig;
using TetConfig = fastfem::TetrahedralizationConfig;

using std::ranges::to, std::ranges::max_element;
using std::views::transform;
using namespace ImGui;

namespace {
constexpr ImVec2 ChartSize{-1, 160};

void WriteWav(std::span<const float> frames, const fs::path &file_path, uint32_t sample_rate, std::optional<float> normalize_max = {}) {
    WavWriter writer{file_path, sample_rate};
    if (!writer.IsOpen()) throw std::runtime_error(std::format("Failed to open wav file {}", file_path.string()));
    const float mult = normalize_max ? *normalize_max / *max_element(frames) : 1.0f;
    const auto frames_normed = frames | transform([mult](float f) { return f * mult; }) | to<std::vector>();
    writer.Write(frames_normed);
}

void PlotFrames(std::span<const float> frames, std::string_view label = "Waveform", std::optional<uint32_t> highlight_frame = {}) {
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

void PlotMagnitudeSpectrum(std::span<const float> frames, uint32_t sample_rate, std::string_view label = "Magnitude spectrum", std::optional<float> highlight_freq = {}) {
    static const float *frames_ptr{frames.data()};
    static FFTData fft{ComputeFft(frames, sample_rate)};
    if (frames.data() != frames_ptr) {
        fft = ComputeFft(frames, sample_rate);
        frames_ptr = frames.data();
    }
    if (ImPlot::BeginPlot(label.data(), ChartSize)) {
        static constexpr float MinDb = -200;
        const uint32_t N = fft.NumReal, N2 = N / 2;
        const auto fs_n = float(sample_rate) / float(N);
        static std::vector<float> frequency(N2), magnitude(N2);
        frequency.resize(N2);
        magnitude.resize(N2);

        for (uint32_t i = 0; i < N2; i++) {
            frequency[i] = fs_n * float(i);
            magnitude[i] = 20.0f * log10f(std::abs(fft.Bins[i]) / float(N2));
        }

        ImPlot::SetupAxes("Frequency (Hz)", "Magnitude (dB)");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, float(sample_rate) / 2, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, MinDb, 0, ImGuiCond_Always);
        if (highlight_freq) {
            ImPlot::PlotInfLines("##Highlight", &(*highlight_freq), 1, {ImPlotProp_LineColor, ImGui::GetStyleColorVec4(ImGuiCol_PlotLinesHovered)});
        }
        ImPlot::PlotShaded("", frequency.data(), magnitude.data(), N2, MinDb, {ImPlotProp_FillColor, ImGui::GetStyleColorVec4(ImGuiCol_PlotHistogramHovered)});
        ImPlot::EndPlot();
    }
}

std::optional<size_t> PlotModeData(
    const std::vector<float> &data, std::string_view label, std::string_view x_label, std::string_view y_label,
    std::optional<size_t> highlight_index = {}, std::optional<float> max_value_opt = {}
) {
    std::optional<size_t> hovered_index;
    if (ImPlot::BeginPlot(label.data(), ChartSize)) {
        static constexpr double BarSize = 0.9;
        const float max_value = max_value_opt.value_or(*std::ranges::max_element(data));
        ImPlot::SetupAxes(x_label.data(), y_label.data());
        ImPlot::SetupAxesLimits(-0.5f, data.size() - 0.5f, 0, max_value, ImPlotCond_Always);
        if (ImPlot::IsPlotHovered()) {
            if (const auto i = std::lround(ImPlot::GetPlotMousePos().x); i >= 0 && i < std::ssize(data)) hovered_index = i;
        }
        if (!highlight_index) {
            ImPlot::PlotBars("", data.data(), data.size(), BarSize);
        } else {
            for (size_t i = 0; i < data.size(); ++i) {
                ImPlot::PlotBars(i == *highlight_index ? "##0" : "", &data[i], 1, BarSize, i);
            }
        }
        ImPlot::EndPlot();
    }

    return hovered_index;
}

bool DrawModalModelActions(
    state::Registry &r, state::Entity e, state::Entity mesh_entity,
    const ModalSolveSettings &settings, const AcousticMaterial &material
) {
    const bool present = r.all_of<ModalModes>(e);
    const bool solving = IsSolving(r, e);
    bool has_action = false;
    if (r.all_of<SoundVerticesModel>(e)) {
        if (Button(present ? "Delete sound object ●" : "Delete sound object")) {
            action::Emit(action::audio::DeleteSoundObject{});
            return true;
        }
        has_action = true;
    }
    if (!present) {
        if (has_action) SameLine();
        if (solving) BeginDisabled();
        if (Button("Create modal model ○")) {
            action::Emit(action::audio::EnsureModalSettings{});
            LaunchModalSolve(r, e, settings, material);
        }
        if (solving) EndDisabled();
    }
    if (present && !solving && r.all_of<ModalSolveSettings, AcousticMaterial>(e) &&
        ModalModelStale(r, e, BuildSolveInputs(r, e, mesh_entity, settings), material)) {
        if (has_action) SameLine();
        if (Button("Update modal model")) LaunchModalSolve(r, e, settings, material);
    }
    return false;
}

void DrawModalModelSettings(
    state::Registry &r, state::Entity e, state::Entity mesh_entity,
    const ModalSolveSettings &settings, const AcousticMaterial &material
) {
    const ContactSurface default_surface = WithPreset({}, surfaces::acoustic::Default);
    const auto &surface = r.all_of<ContactSurface>(e) ? r.get<const ContactSurface>(e) : default_surface;
    ui::PatchEdit fs{settings};

    SeparatorText("Material properties");
    ui::ChoiceCombo("Presets", material.Name, materials::acoustic::All | transform(&AcousticMaterial::Name), std::identity{}, [&](const std::string &name) {
        action::Emit(action::audio::SetMaterialPreset{name});
    });
    ui::DrawFields<AcousticMaterialProperties>(ui::PatchEdit{material}.Sub<&AcousticMaterial::Properties>());

    DrawContactSurfaceControls(r, e, surface, material);

    SeparatorText("Modes");
    ui::DrawFields<ModalSolverConfig>(fs.Sub<&ModalSolveSettings::Solve, &SurfaceSolveConfig::Modal>());

    SeparatorText("Discretization");
    fs.Draw<&ModalSolveSettings::Discretization>();
    auto solve = fs.Sub<&ModalSolveSettings::Solve>();
    if (settings.Discretization == fastfem::Discretization::Tet10) {
        SeparatorText(field::ChainLabel<&SurfaceSolveConfig::Tetrahedralization>.c_str());
        auto tet = solve.Sub<&SurfaceSolveConfig::Tetrahedralization>();
        ui::DrawFields<TetConfig, &TetConfig::Holes>(tet);
        solve.Draw<&SurfaceSolveConfig::SurfaceSimplificationRatio>();
        if (settings.Solve.Tetrahedralization.Refinement == fastfem::TetRefinement::QualityAndResolution) solve.Draw<&SurfaceSolveConfig::Resolution>();
        tet.Draw<&TetConfig::Holes>([](const char *label, std::vector<dvec3> &holes) {
            BeginGroup();
            bool changed = false;
            for (size_t i = 0; i < holes.size(); ++i) {
                PushID(int(i));
                changed |= InputScalarN(label, ImGuiDataType_Double, &holes[i].x, 3);
                SameLine();
                if (Button("Remove")) {
                    holes.erase(holes.begin() + i);
                    changed = true;
                    PopID();
                    break;
                }
                PopID();
            }
            if (Button("Add hole seed")) {
                holes.emplace_back();
                changed = true;
            }
            EndGroup();
            return changed;
        });
    } else {
        ui::DrawFields<SurfaceSolveConfig, &SurfaceSolveConfig::Modal, &SurfaceSolveConfig::Tetrahedralization, &SurfaceSolveConfig::SurfaceSimplificationRatio>(solve);
    }

    SeparatorText("Excitation vertices");
    const uint32_t num_vertices = GetMesh(r, mesh_entity).VertexCount();
    const bool has_excitable = r.all_of<SoundVertices>(e);
    const bool reuse = has_excitable && settings.CopySoundVertices;
    if (has_excitable) fs.Draw<&ModalSolveSettings::CopySoundVertices>();
    if (reuse) BeginDisabled();
    fs.Draw<&ModalSolveSettings::NumVertices>([&](const char *label, uint32_t &count) {
        const uint32_t min_vertices = 1, max_vertices = num_vertices;
        count = reuse ? r.get<const SoundVertices>(e).Vertices.Count : std::clamp(count, min_vertices, max_vertices);
        return SliderScalar(label, ImGuiDataType_U32, &count, &min_vertices, &max_vertices);
    });
    if (reuse) EndDisabled();
}

// Returns the active vertex in Excite mode or selected vertices in Edit mode.
std::vector<uint32_t> GetSampleOpVertices(const state::Registry &r, state::Entity sound_entity) {
    if (!r.valid(sound_entity)) return {};
    const auto *inst = r.try_get<const Instance>(sound_entity);
    if (!inst) return {};
    const auto mesh_entity = inst->Entity;
    const auto mesh = TryGetMesh(r, mesh_entity);
    if (!mesh) return {};

    const auto mode = r.get<const Interaction>(state::Document).Mode;
    if (mode == InteractionMode::Excite) {
        if (const auto *active = r.try_get<const MeshActiveElement>(mesh_entity)) return {active->Handle};
        return {};
    }
    if (mode != InteractionMode::Edit || !r.all_of<MeshElementSelection>(mesh_entity)) return {};

    const auto selected = r.Resources.get<const MeshStore>().GetSelectedElements(mesh->GetStoreId(), Element::Vertex);
    std::vector<uint32_t> vertices;
    selected.ForEach([&](uint32_t vertex) { vertices.push_back(mesh->VertexOrdinal(he::VH{vertex})); });
    return vertices;
}

// Circular pad returning a position in the unit disk (center = zero).
// Drag to set, right-click recenters.
bool ImpulseJoystick(vec2 &pos) {
    constexpr float radius{32.f};
    const auto p0 = GetCursorScreenPos();
    InvisibleButton("impulse", {radius * 2, radius * 2}, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    const ImVec2 center{p0.x + radius, p0.y + radius};
    bool changed = false;
    if (IsItemActive() && IsMouseDown(ImGuiMouseButton_Left)) {
        const auto m = GetIO().MousePos;
        pos = {(m.x - center.x) / radius, -(m.y - center.y) / radius};
        if (const float len = Length(pos); len > 1.f) pos /= len;
        changed = true;
    } else if (IsItemClicked(ImGuiMouseButton_Right)) {
        pos = {0, 0};
        changed = true;
    }
    auto &dl = *GetWindowDrawList();
    dl.AddCircleFilled(center, radius, GetColorU32(ImGuiCol_FrameBg));
    dl.AddCircle(center, radius, GetColorU32(ImGuiCol_Border));
    dl.AddCircleFilled({center.x + pos.x * radius, center.y - pos.y * radius}, 4.f, GetColorU32(IsItemActive() ? ImGuiCol_SliderGrabActive : ImGuiCol_SliderGrab));
    return changed;
}
} // namespace

void DrawObjectAudioControls(state::Registry &r, state::Entity e, state::Entity mesh_entity) {
    if (e == state::Null || mesh_entity == state::Null) return;

    const ModalSolveSettings default_settings;
    const auto &settings = r.all_of<ModalSolveSettings>(e) ? r.get<const ModalSolveSettings>(e) : default_settings;
    const auto &material = r.all_of<AcousticMaterial>(e) ? r.get<const AcousticMaterial>(e) : materials::acoustic::All.front();
    if (DrawModalModelActions(r, e, mesh_entity, settings, material)) return;

    // Sample ops (Add/Replace/Remove) are only available in Edit / Excite mode.
    const auto mode = r.get<const Interaction>(state::Document).Mode;
    const bool sample_ops_available = mode == InteractionMode::Edit || mode == InteractionMode::Excite;
    const auto op_vertices = sample_ops_available ? GetSampleOpVertices(r, e) : std::vector<uint32_t>{};

    const bool has_model = r.all_of<SoundVerticesModel>(e);
    if (!has_model) DrawModalModelSettings(r, e, mesh_entity, settings, material);

    const auto *samples = r.try_get<const VertexSamples>(e);
    const auto *modal_modes = r.try_get<const ModalModes>(e);
    const auto *excitable = r.try_get<const SoundVertices>(e);
    auto model = has_model ? r.get<SoundVerticesModel>(e) : SoundVerticesModel::Samples;
    const auto *recording = r.try_get<const Recording>(e);
    const uint32_t active_vi = excitable ? GetActiveVertexIndex(r, e) : 0;
    const auto sample_rate = DeviceSampleRate(r); // for the spectrum plots below

    if (samples && modal_modes) {
        PushID("SelectAudioModel");
        auto edit_model = int(model);
        bool model_changed = RadioButton("Recordings", &edit_model, int(SoundVerticesModel::Samples));
        SameLine();
        model_changed |= RadioButton("Modal", &edit_model, int(SoundVerticesModel::Modal));
        PopID();
        if (model_changed) {
            model = SoundVerticesModel(edit_model);
            action::Emit(action::audio::SetModel{model});
        }
    }

    if (has_model && excitable) {
        const auto excitable_vertices = r.Resources.get<const MeshStore>().Arenas().SoundVertices.Get(excitable->Vertices);
        const auto active_vertex = excitable_vertices[active_vi];
        const bool can_excite =
            (model == SoundVerticesModel::Samples) ||
            (model == SoundVerticesModel::Modal && (!recording || recording->Complete()));
        if (!can_excite) BeginDisabled();
        Button("Excite");
        if (IsItemActivated()) action::Emit(action::audio::StartExcite{active_vertex});
        else if (IsItemDeactivated()) action::Emit(action::audio::StopExcite{});
        if (!can_excite) EndDisabled();
        SameLine();
        if (BeginCombo("Vertex", std::to_string(active_vertex).c_str())) {
            for (uint32_t vi = 0; vi < excitable_vertices.size(); ++vi) {
                if (const auto vertex = excitable_vertices[vi]; Selectable(std::to_string(vertex).c_str(), vi == active_vi)) {
                    action::Emit(action::audio::SetExciteVertex{vi, vertex});
                }
            }
            EndCombo();
        }

        if (model == SoundVerticesModel::Modal) {
            TextUnformatted("Impact angle");
            ImpulseJoystick(ImpulseAngle);
            ui::Describe("Impact angle", "Strike direction relative to the surface, from perpendicular at the center to tangent at the edge. Right-click to recenter.");
        }
    }

    if (model == SoundVerticesModel::Samples) {
        if (has_model) SeparatorText("Sound samples");
        if (sample_ops_available) {
            std::vector<uint32_t> op_with_sample;
            if (samples) {
                for (const uint32_t mv : op_vertices) {
                    if (samples->PathByVertex.contains(mv)) op_with_sample.push_back(mv);
                }
            }
            const auto n = op_vertices.size(), with_sample = op_with_sample.size();
            if (n == 0) BeginDisabled();
            if (const auto assign_label = n > 1 ? std::format("Assign sample to {} vertices…", n) : std::string{with_sample ? "Replace sample…" : "Assign sample…"};
                Button(assign_label.c_str())) {
                FileDialog::ShowOpen("wav;mp3;flac;ogg;opus", [verts = op_vertices](const fs::path &path) mutable {
                    action::Emit(action::audio::AssignVertexSamples{std::make_unique<std::vector<uint32_t>>(std::move(verts)), path});
                });
            }
            if (n == 0) EndDisabled();
            if (with_sample > 0) {
                SameLine();
                if (const auto remove_label = with_sample > 1 ? std::format("Remove {} samples", with_sample) : std::string{"Remove sample"};
                    Button(remove_label.c_str())) {
                    action::Emit(action::audio::RemoveVertexSamples{std::move(op_with_sample)});
                    return;
                }
            }
        }
        if (const auto path = ActiveSamplePath(r, e)) {
            const auto &frames = GetSampleFrames(r, *path);
            if (!frames.empty()) {
                const auto *playback = r.try_get<const SamplePlayback>(e);
                PlotFrames(frames, "Waveform", !playback || playback->Stopped ? std::optional<uint32_t>{} : std::optional{playback->Frame});
                PlotMagnitudeSpectrum(frames, sample_rate, "Spectrum");
            }
        }
    }

    if (!has_model) return;

    DrawModalModelSettings(r, e, mesh_entity, settings, material);

    if (model != SoundVerticesModel::Modal) return;

    if (!excitable || !modal_modes) return;

    static std::optional<size_t> hovered_mode_index;
    const auto &modes = *modal_modes;
    if (recording && recording->Complete()) {
        const auto &frames = recording->Frames;
        PlotFrames(frames, "Modal impact waveform");
        const auto highlight_freq = hovered_mode_index ? std::optional{modes.Freqs[*hovered_mode_index]} : std::nullopt;
        PlotMagnitudeSpectrum(frames, sample_rate, "Modal impact spectrum", highlight_freq);
    }

    if (CollapsingHeader("Modal data charts")) {
        std::optional<size_t> new_hovered_index;
        if (const auto hovered = PlotModeData(modes.Freqs, "Mode frequencies", "", "Frequency (Hz)", hovered_mode_index)) new_hovered_index = hovered;
        if (const auto hovered = PlotModeData(modes.T60s, "Mode T60s", "", "T60 decay time (s)", hovered_mode_index)) new_hovered_index = hovered;
        const auto active_gains = [&]() -> std::vector<float> {
            if (active_vi >= modes.Shapes.size()) return {};
            const auto j = TiltAlongNormal(VertexNormal(GetMesh(r, mesh_entity), r.Resources.get<const MeshStore>().Arenas().SoundVertices.Get(excitable->Vertices)[active_vi]), ImpulseAngle);
            return modes.Shapes[active_vi] | transform([&](const vec3 &s) { return std::abs(Dot(s, j)); }) | to<std::vector<float>>();
        }();
        if (!active_gains.empty()) {
            if (const auto hovered = PlotModeData(active_gains, "Mode gains", "Mode index", "Gain", hovered_mode_index)) new_hovered_index = hovered;
        }
        if (hovered_mode_index = new_hovered_index; hovered_mode_index && *hovered_mode_index < modes.Freqs.size()) {
            const auto index = *hovered_mode_index;
            Text(
                "Mode %lu: Freq (scaled) %.2f Hz, Freq (FEM) %.2f, T60 %.2f s, Gain %.4f", index,
                modes.Freqs[index],
                modes.Freqs[index] * modes.OriginalFundamentalFreq / modes.Freqs[0],
                modes.T60s[index],
                index < active_gains.size() ? active_gains[index] : 0.f
            );
        }
    }

    if (CollapsingHeader("Synthesis")) {
        ui::Edit fe{r, e};
        ui::DrawFields<ModalGain>(fe);
        ui::DrawFields<ModalTuning>(fe);
    }

    const bool is_recording = recording && !recording->Complete();
    if (is_recording) BeginDisabled();
    static constexpr uint32_t RecordFrames = 208'592; // Same length as RealImpact recordings.
    if (Button("Record strike")) action::Emit(action::audio::StartRecording{RecordFrames});
    if (is_recording) EndDisabled();

    if (samples && recording && recording->Complete()) {
        SameLine();
        if (Button("Save wav files")) {
            const auto name = GetName(r, e);
            static const auto WavOutDir = fs::path{".."} / "audio_samples";
            const auto sr = DeviceSampleRate(r);
            WriteWav(recording->Frames, WavOutDir / std::format("{}-modal", name), sr);
            if (const auto path = ActiveSamplePath(r, e)) {
                WriteWav(GetSampleFrames(r, *path), WavOutDir / std::format("{}-impact", name), sr);
            }
        }
    }
}

void DrawGlobalSynthControls(state::Registry &r) {
    ui::Edit f{r, state::Document};
    if (!r.view<const ModalModes>().empty() && CollapsingHeader("Modal synthesis", ImGuiTreeNodeFlags_DefaultOpen)) {
        ui::DrawFields<ModalSoundControls, &ModalSoundControls::SampleGain>(f);

        DrawSurfaceSynthControls(r);

        SeparatorText("Striker");
        const auto &striker = r.get<const Striker>(state::Document);
        ui::ChoiceCombo("Material", striker.Material.Name, materials::acoustic::All | transform(&AcousticMaterial::Name), std::identity{}, [&](const std::string &name) {
            action::Emit(action::audio::SetMaterialPreset{name, true});
        });
        ui::DrawFields<Striker, &Striker::Material>(f);
        Text("Mass: %.3g kg", StrikerMass(striker));
        ui::Describe("Striker mass", "Mass of the capsule mallet that strikes objects. A harder material or lighter capsule brightens the contact.");
    }
    if (!r.view<const VertexSamples>().empty() && CollapsingHeader("Samples", ImGuiTreeNodeFlags_DefaultOpen)) {
        f.Draw<&ModalSoundControls::SampleGain>();
    }
}

void DrawAudioDebug(const state::Registry &r) {
    const auto &m = r.Resources.get<const ModalAudio>();
    const auto &bank = *m.Live;

    SeparatorText("Device");
    if (const auto *device = r.Resources.find<AudioDeviceResource>(); device && device->Initialized) {
        Text("%s at %u Hz", device->DeviceName.empty() ? "System default" : device->DeviceName.c_str(), device->SampleRate);
    } else {
        TextUnformatted("No output device");
    }

    SeparatorText("Modal bank");
    Text("Objects: %zu, modes: %zu", bank.Entities.size(), bank.CoeffRe.size());
    Text("Impacts: %u / %u", m.ActiveImpacts.load(std::memory_order_relaxed), m.MaxImpacts.load(std::memory_order_relaxed));
    Text("Energy: %.3g J, peak %.3g J", m.ModalEnergy.load(std::memory_order_relaxed), m.PeakModalEnergy.load(std::memory_order_relaxed));
    ui::Describe("Modal energy", "Mechanical energy in the mode banks, in joules. A peak that rises between strikes means a channel is feeding the modes.");

    SeparatorText("Render");
    const auto share = m.RenderShare.load(std::memory_order_relaxed);
    static constexpr const char *RenderHelp{"Render time per audio block as a share of the block's duration. Above 100% the device underruns."};
    Text("Block: %.2f ms (%.0f%% of budget)", m.RenderSeconds.load(std::memory_order_relaxed) * 1e3f, share * 100);
    ui::Describe("Render block", RenderHelp);
    Text("Worst: %.0f%%", m.PeakRenderShare.load(std::memory_order_relaxed) * 100);
    ui::Describe("Worst render block", RenderHelp);

    SeparatorText("Event queue");
    const auto queued = m.EventWrite.load(std::memory_order_relaxed) - m.EventRead.load(std::memory_order_acquire);
    Text("Queued: %u / %u", queued, ModalAudio::EventCapacity);
    Text("Dropped: %llu", m.EventsDropped);
    ui::Describe("Dropped events", "Strike and contact events dropped because the queue was full");

    DrawSurfaceContactDebug(r);
}

namespace {
std::string_view SolveStageLabel(fastfem::SolveStage stage) {
    using enum fastfem::SolveStage;
    switch (stage) {
        case PreparingSurface: return "Preparing surface";
        case GeneratingTetrahedra: return "Generating tetrahedra";
        case BuildingFiniteCellGrid: return "Building finite-cell grid";
        case ComputingMassProperties: return "Computing mass properties";
        case BuildingTopology: return "Building topology";
        case AssemblingOperators: return "Assembling operators";
        case Factorizing: return "Factorizing";
        case SolvingEigenproblem: return "Solving eigenproblem";
        case SamplingModes: return "Sampling modes";
        case Finalizing: return "Finalizing modal model";
        case Complete: return "Complete";
    }
}
} // namespace

void DrawModalJobsOverlay(state::Registry &r) {
    const auto &jobs = r.Resources.get<const ModalSolveJobs>().Jobs;
    if (jobs.empty()) return;

    constexpr float Pad{12.f};
    const auto anchor = GetWindowPos() + ImVec2{Pad, GetWindowSize().y - Pad};
    SetNextWindowPos(anchor, ImGuiCond_Always, {0.f, 1.f});
    SetNextWindowBgAlpha(0.85f);
    // The viewport panel zeroes its window padding, so restore normal padding for the overlay.
    // Vertical padding matches the item spacing, so each row sits evenly between the panel edges and its bar.
    PushStyleVar(ImGuiStyleVar_WindowPadding, {10.f, 6.f});
    PushStyleVar(ImGuiStyleVar_WindowRounding, 6.f);
    PushStyleVar(ImGuiStyleVar_ItemSpacing, {GetStyle().ItemSpacing.x, 6.f});
    PushStyleVar(ImGuiStyleVar_FramePadding, {8.f, 3.f});
    constexpr ImGuiWindowFlags OverlayFlags =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoMove;
    if (Begin("Modal solve jobs", nullptr, OverlayFlags)) {
        // Keep the overlay above the focused viewport panel.
        BringWindowToDisplayFront(GetCurrentWindow());
        for (const auto &job : jobs) {
            const auto &monitor = *job->Work.Monitor;
            PushID(job.get());
            BeginGroup();
            AlignTextToFramePadding();
            ImSpinner::SpinnerRotateSegmentsPulsar("##spinner", GetTextLineHeight() * 0.5f, 2.f, GetColorU32(ImGuiCol_Text), 1.1f, 3, 3);
            SameLine();
            const auto stage = SolveStageLabel(monitor.Stage.load(std::memory_order_relaxed));
            Text("%s: %.*s", job->Work.Title.c_str(), int(stage.size()), stage.data());
            SameLine(0.f, GetStyle().ItemSpacing.x * 3.f);
            const bool cancelled = job->Work.Cancelled();
            if (cancelled) BeginDisabled();
            if (Button("Cancel")) job->Work.RequestCancel();
            if (cancelled) EndDisabled();
            EndGroup();
            const float progress = monitor.Progress.load(std::memory_order_relaxed);
            ProgressBar(progress > 0.f ? progress : -float(GetTime()), {GetItemRectSize().x, 3.f}, "");
            PopID();
        }
    }
    End();
    PopStyleVar(4);
}
