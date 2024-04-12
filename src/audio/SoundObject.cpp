#include "SoundObject.h"

#include <format>

#include "imgui.h"

#include "mesh2faust.h"

using Sample = float;
#ifndef FAUSTFLOAT
#define FAUSTFLOAT Sample
#endif

#include "faust/dsp/llvm-dsp.h"

#include "FaustParams.h"

#include "tetMesh.h" // Vega
#include "tetgen.h" // Must be after any Faust includes, since it defined a `REAL` macro.

#include "Tets.h"
#include "Worker.h"

using std::string, std::string_view;

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

                    uint sample_rate = 48000; // todo follow device sample rate
                    Dsp->init(sample_rate);
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

SoundObjectData::Modal::Modal() : FaustDsp(std::make_unique<FaustDSP>()) {}
SoundObjectData::Modal::~Modal() = default;

// Worker DspGenerator{"Generate DSP code", "Generating DSP code..."};

string GenerateDsp(const tetgenio &tets, const MaterialProperties &material, const std::vector<uint> &excitable_vertex_indices, bool freq_control = false) {
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

    std::vector<int> excitable_vertex_indices_ints; // Convert to signed integers.
    excitable_vertex_indices_ints.reserve(excitable_vertex_indices.size());
    for (uint i : excitable_vertex_indices) excitable_vertex_indices_ints.emplace_back(i);

    m2f::CommonArguments args{
        model_name,
        freq_control, // freqency control activated
        20, // lowest mode freq
        10000, // highest mode freq
        40, // number of synthesized modes (default is 20)
        80, // number of modes to be computed for the finite element analysis (default is 100)
        excitable_vertex_indices_ints, // specific excitation positions
        int(excitable_vertex_indices.size()), // number of excitation positions (default is max: -1)
    };

    const auto m2f_result = m2f::mesh2faust(&volumetric_mesh, args);
    const string model_dsp = m2f_result.modelDsp;
    if (model_dsp.empty()) return "process = 0;";

    const auto &mode_freqs = m2f_result.model.modeFreqs;
    const float fundamental_freq = mode_freqs.empty() ? 440.0f : mode_freqs.front();

    // Static code sections.
    static const string to_sandh = " : ba.sAndH(gate);"; // Add a sample and hold on the gate, in serial, and end the expression.
    static const string
        gain = "gain = hslider(\"gain[scale:log]\",0.1,0,0.5,0.01);",
        t60_scale = "t60Scale = hslider(\"t60[scale:log][tooltip: Resonance duration (s) of the lowest mode.]\",16,0,50,0.01)" + to_sandh,
        t60_decay = "t60Decay = hslider(\"t60 Decay[scale:log][tooltip: Decay of modes as a function of their frequency, in t60 units.\nAt 1, the t60 of the highest mode will be close to 0 seconds.]\",0.80,0,1,0.01)" + to_sandh,
        t60_slope = "t60Slope = hslider(\"t60 Slope[scale:log][tooltip: Power of the function used to compute the decay of modes t60 in function of their frequency.\nAt 1, decay is linear. At 2, decay slope has degree 2, etc.]\",2.5,1,6,0.01)" + to_sandh,
        source = "source = vslider(\"Excitation source [style:radio {'Hammer':0;'Audio input':1 }]\",0,0,1,1);",
        gate = "gate = button(\"gate[tooltip: When excitation source is 'Hammer', excites the vertex. With any excitation source, applies the current parameters.]\");",
        hammer_hardness = "hammerHardness = hslider(\"hammerHardness[tooltip: Only has an effect when excitation source is 'Hammer'.]\",0.9,0,1,0.01)" + to_sandh,
        hammer_size = "hammerSize = hslider(\"hammerSize[tooltip: Only has an effect when excitation source is 'Hammer'.]\",0.3,0,1,0.01)" + to_sandh,
        hammer = "hammer(trig,hardness,size) = en.ar(att,att,trig)*no.noise : fi.lowpass(3,ctoff)\nwith{ ctoff = (1-size)*9500+500; att = (1-hardness)*0.01+0.001; };";

    // Variable code sections.
    const uint num_excite_pos = excitable_vertex_indices.size();
    const string
        freq = std::format("freq = hslider(\"Frequency[scale:log][tooltip: Fundamental frequency of the model]\",{},60,8000,1){}", fundamental_freq, to_sandh),
        ex_pos = std::format("exPos = nentry(\"exPos\",{},0,{},1){}", (num_excite_pos - 1) / 2, num_excite_pos - 1, to_sandh),
        modal_model = std::format("{}({}exPos,t60Scale,t60Decay,t60Slope)", model_name, freq_control ? "freq," : ""),
        process = std::format("process = hammer(gate,hammerHardness,hammerSize),_ : select2(source) : {}*gain;", modal_model);

    std::stringstream instrument;
    instrument << source << '\n'
               << gate << '\n'
               << hammer_hardness << '\n'
               << hammer_size << '\n'
               << gain << '\n'
               << freq << '\n'
               << ex_pos << '\n'
               << t60_scale << '\n'
               << t60_decay << '\n'
               << t60_slope << '\n'
               << '\n'
               << hammer << '\n'
               << '\n'
               << process << '\n';

    return model_dsp + instrument.str();
}

SoundObject::SoundObject(const ::Tets &tets, MaterialProperties &&material, vec3 listener_position)
    : Tets(tets), ListenerPosition(std::move(listener_position)), Material(std::move(material)), ModalData(std::in_place) {}

SoundObject::SoundObject(const ::Tets &tets, MaterialProperties &&material, vec3 listener_position, std::unordered_map<uint, std::vector<float>> &&impace_frames_by_vertex)
    : Tets(tets), ListenerPosition(std::move(listener_position)), Material(std::move(material)),
      ImpactAudioData({std::move(impace_frames_by_vertex)}), ModalData(std::in_place) {
    for (auto &[vertex, _] : ImpactAudioData->ImpactFramesByVertex) ExcitableVertices.emplace_back(vertex);
    CurrentVertex = ExcitableVertices.front();
}

SoundObject::~SoundObject() = default;

void SoundObject::ProduceAudio(DeviceData device, float *input, float *output, uint frame_count) {
    if (Model == SoundObjectModel::ImpactAudio && ImpactAudioData) {
        if (!ImpactAudioData->ImpactFramesByVertex.contains(CurrentVertex)) return;

        const auto &impact_samples = ImpactAudioData->ImpactFramesByVertex.at(CurrentVertex);
        const uint sample_rate = device.SampleRate; // todo - resample from 48kHz to device sample rate if necessary
        (void)sample_rate; // Unused

        for (uint i = 0; i < frame_count; ++i) {
            output[i] += ImpactAudioData->CurrentFrame < impact_samples.size() ? impact_samples[ImpactAudioData->CurrentFrame++] : 0.0f;
        }
    } else if (Model == SoundObjectModel::Modal && ModalData) {
        ModalData->FaustDsp->Compute(frame_count, &input, &output);
    }
}

void SoundObject::Strike(float force) {
    if (ImpactAudioData) {
        ImpactAudioData->CurrentFrame = 0;
    }
    if (ModalData) {
        // todo
    }
    (void)force; // Unused
}

void SoundObject::SetModel(SoundObjectModel model) {
    Model = model;
}

using namespace ImGui;

void SoundObject::RenderControls() {
    if (Button("Strike")) {
        Strike();
    }
    PushID("AudioModel");
    int model = int(Model);
    bool model_changed = RadioButton("Recordings", &model, int(SoundObjectModel::ImpactAudio));
    SameLine();
    model_changed |= RadioButton("Modal", &model, int(SoundObjectModel::Modal));
    PopID();
    if (model_changed) SetModel(SoundObjectModel(model));
    if (Model == SoundObjectModel::ImpactAudio && ImpactAudioData) {
        if (BeginCombo("Vertex", std::to_string(CurrentVertex).c_str())) {
            for (auto &[vertex, _] : ImpactAudioData->ImpactFramesByVertex) {
                if (Selectable(std::to_string(vertex).c_str(), vertex == CurrentVertex)) CurrentVertex = vertex;
            }
            EndCombo();
        }
    } else if (Model == SoundObjectModel::Modal && ModalData) {
        SeparatorText("Material properties");

        static std::string selected_preset = DefaultMaterialPresetName;
        if (BeginCombo("Presets", selected_preset.c_str())) {
            for (const auto &[preset_name, material] : MaterialPresets) {
                const bool is_selected = (preset_name == selected_preset);
                if (Selectable(preset_name.c_str(), is_selected)) {
                    selected_preset = preset_name;
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

        if (Button("Generate DSP")) {
            if (!ImpactAudioData) {
                // ImpactAudio objects can only be struck at the impact points.
                // Otherwise, linearly distribute the vertices across the tet mesh.
                ExcitableVertices.clear();
                const uint num_excitable_vertices = 5; // todo UI input
                ExcitableVertices.reserve(num_excitable_vertices);
                for (uint i = 0; i < num_excitable_vertices; ++i) {
                    ExcitableVertices.emplace_back(i * (*Tets).numberofpoints / num_excitable_vertices);
                }
            }
            const bool freq_control = true;
            ModalData->FaustDsp->SetCode(GenerateDsp(*Tets, Material, ExcitableVertices, freq_control));
        }
        if (ModalData->FaustDsp->Ui) ModalData->FaustDsp->Ui->Draw();
    }
}
