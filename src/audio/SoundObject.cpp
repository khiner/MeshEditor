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

#include "tetgen.h" // Must be after any Faust includes, since it defined a `REAL` macro.

#include "RealImpact.h"
#include "Worker.h"
#include "mesh/Mesh.h"

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

SoundObjectData::Modal::Modal(const ::Mesh &mesh) : Mesh(mesh), FaustDsp(std::make_unique<FaustDSP>()) {}
SoundObjectData::Modal::~Modal() = default;

// Worker DspGenerator{"Generate DSP code", "Generating DSP code..."};
// Worker TetGenerator{"Generate tet mesh", "Generating tetrahedral mesh...", [&] { GenerateTets(); }};
std::unique_ptr<tetgenio> TetGenResult;

tetgenio GenerateTets(const Mesh &mesh, bool quality = false) {
    tetgenio in;
    in.firstnumber = 0;
    const float *vertices = mesh.GetPositionData();
    const auto triangle_indices = mesh.CreateTriangleIndices();
    in.numberofpoints = mesh.GetVertexCount();
    in.pointlist = new REAL[in.numberofpoints * 3];
    for (uint i = 0; i < uint(in.numberofpoints); ++i) {
        in.pointlist[i * 3] = vertices[i * 3];
        in.pointlist[i * 3 + 1] = vertices[i * 3 + 1];
        in.pointlist[i * 3 + 2] = vertices[i * 3 + 2];
    }

    in.numberoffacets = triangle_indices.size() / 3;
    in.facetlist = new tetgenio::facet[in.numberoffacets];

    for (uint i = 0; i < uint(in.numberoffacets); ++i) {
        tetgenio::facet &f = in.facetlist[i];
        tetgenio::init(&f);
        f.numberofpolygons = 1;
        f.polygonlist = new tetgenio::polygon[f.numberofpolygons];
        f.polygonlist[0].numberofvertices = 3;
        f.polygonlist[0].vertexlist = new int[f.polygonlist[0].numberofvertices];
        f.polygonlist[0].vertexlist[0] = triangle_indices[i * 3];
        f.polygonlist[0].vertexlist[1] = triangle_indices[i * 3 + 1];
        f.polygonlist[0].vertexlist[2] = triangle_indices[i * 3 + 2];
    }

    const string options = quality ? "pq" : "p";
    std::vector<char> options_mutable(options.begin(), options.end());
    tetgenio result;
    tetrahedralize(options_mutable.data(), &in, &result);

    return result;
}

string GenerateDsp(const tetgenio &tets, const MaterialProperties &material, const std::vector<int> &excitable_vertex_indices, bool freq_control = false) {
    std::vector<int> tet_indices;
    tet_indices.reserve(tets.numberoftetrahedra * 4 * 3); // 4 triangles per tetrahedron, 3 indices per triangle.
    // Turn each tetrahedron into 4 triangles.
    for (uint i = 0; i < uint(tets.numberoftetrahedra); ++i) {
        auto &result_indices = tets.tetrahedronlist;
        uint tri_i = i * 4;
        int a = result_indices[tri_i], b = result_indices[tri_i + 1], c = result_indices[tri_i + 2], d = result_indices[tri_i + 3];
        tet_indices.insert(tet_indices.end(), {a, b, c, d, a, b, c, d, a, b, c, d});
    }
    // Convert the tetrahedral mesh into a VegaFEM Tets.
    TetMesh volumetric_mesh{
        tets.numberofpoints, tets.pointlist, tets.numberoftetrahedra * 3, tet_indices.data(),
        material.YoungModulus, material.PoissonRatio, material.Density
    };

    static const string model_name = "modalModel";

    m2f::CommonArguments args{
        model_name,
        freq_control, // freqency control activated
        20, // lowest mode freq
        10000, // highest mode freq
        40, // number of synthesized modes (default is 20)
        80, // number of modes to be computed for the finite element analysis (default is 100)
        excitable_vertex_indices, // specific excitation positions
        int(excitable_vertex_indices.size()), // number of excitation positions (default is max: -1)
    };

    const string model_dsp = m2f::mesh2faust(&volumetric_mesh, args);
    if (model_dsp.empty()) return "process = 0;";

    // todo nextup: a new `mesh2faust` response with the model dsp and metadata (modes, ...), and use that to compute the default frequency.
    //   (or add the default frequency directly to the metadata)
    const float default_freq = 220;

    // Static code sections.
    static const string
        gain = "gain = hslider(\"gain[scale:log]\",0.1,0,0.5,0.01);",
        t60_scale = "t60Scale = hslider(\"t60[scale:log][tooltip: Resonance duration (s) of the lowest mode.]\",16,0,50,0.01) : ba.sAndH(gate);",
        t60_decay = "t60Decay = hslider(\"t60 Decay[scale:log][tooltip: Decay of modes as a function of their frequency, in t60 units.\nAt 1, the t60 of the highest mode will be close to 0 seconds.]\",0.80,0,1,0.01) : ba.sAndH(gate);",
        t60_slope = "t60Slope = hslider(\"t60 Slope[scale:log][tooltip: Power of the function used to compute the decay of modes t60 in function of their frequency.\nAt 1, decay is linear. At 2, decay slope has degree 2, etc.]\",2.5,1,6,0.01) : ba.sAndH(gate);",
        source = "source = vslider(\"Excitation source [style:radio {'Hammer':0;'Audio input':1 }]\",0,0,1,1);",
        gate = "gate = button(\"gate[tooltip: When excitation source is 'Hammer', excites the vertex. With any excitation source, applies the current parameters.]\");",
        hammer_hardness = "hammerHardness = hslider(\"hammerHardness[tooltip: Only has an effect when excitation source is 'Hammer'.]\",0.9,0,1,0.01) : ba.sAndH(gate);",
        hammer_size = "hammerSize = hslider(\"hammerSize[tooltip: Only has an effect when excitation source is 'Hammer'.]\",0.3,0,1,0.01) : ba.sAndH(gate);",
        hammer = "hammer(trig,hardness,size) = en.ar(att,att,trig)*no.noise : fi.lowpass(3,ctoff)\nwith{ ctoff = (1-size)*9500+500; att = (1-hardness)*0.01+0.001; };";

    // Variable code sections.
    const uint num_excite_pos = excitable_vertex_indices.size();
    const string
        freq = std::format("freq = hslider(\"Frequency[scale:log][tooltip: Fundamental frequency of the model]\",{},60,8000,1) : ba.sAndH(gate);", default_freq),
        ex_pos = std::format("exPos = nentry(\"exPos\",{},0,{},1) : ba.sAndH(gate);", (num_excite_pos - 1) / 2, num_excite_pos - 1),
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

SoundObject::SoundObject(const RealImpact &real_impact, const RealImpactListenerPoint &listener_point)
    : ListenerPosition(listener_point.GetPosition()), RealImpactData(listener_point.LoadImpactSamples(real_impact)) {}

SoundObject::SoundObject(const ::Mesh &mesh, vec3 listener_position)
    : ListenerPosition(listener_position), ModalData(mesh) {
}
SoundObject::SoundObject(const RealImpact &real_impact, const RealImpactListenerPoint &listener_point, const Mesh &mesh)
    : ListenerPosition(listener_point.GetPosition()), RealImpactData(listener_point.LoadImpactSamples(real_impact)), ModalData(mesh) {}

SoundObject::~SoundObject() = default;

void SoundObject::ProduceAudio(DeviceData device, float *input, float *output, uint frame_count) {
    if (Model == SoundObjectModel::RealImpact && RealImpactData) {
        if (RealImpactData->CurrentVertexIndex >= RealImpactData->ImpactSamples.size()) return;

        const uint sample_rate = device.SampleRate; // todo - resample from 48kHz to device sample rate if necessary
        (void)sample_rate; // Unused

        const auto &impact_samples = RealImpactData->ImpactSamples[RealImpactData->CurrentVertexIndex];
        for (uint i = 0; i < frame_count; ++i) {
            output[i] += RealImpactData->CurrentFrame < impact_samples.size() ? impact_samples[RealImpactData->CurrentFrame++] : 0.0f;
        }
    } else if (Model == SoundObjectModel::Modal && ModalData) {
        ModalData->FaustDsp->Compute(frame_count, &input, &output);
    }
}

void SoundObject::Strike(float force) {
    if (RealImpactData) {
        RealImpactData->CurrentFrame = 0;
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
    bool model_changed = RadioButton("RealImpact", &model, int(SoundObjectModel::RealImpact));
    SameLine();
    model_changed |= RadioButton("Modal", &model, int(SoundObjectModel::Modal));
    PopID();
    if (model_changed) SetModel(SoundObjectModel(model));
    if (Model == SoundObjectModel::RealImpact && RealImpactData) {
        if (BeginCombo("Vertex", std::to_string(RealImpactData->CurrentVertexIndex).c_str())) {
            for (uint i = 0; i < RealImpactData->ImpactSamples.size(); ++i) {
                if (Selectable(std::to_string(i).c_str(), i == RealImpactData->CurrentVertexIndex)) {
                    RealImpactData->CurrentVertexIndex = i;
                }
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

        Text("Young's modulus (Pa)");
        InputDouble("##Young's modulus", &Material.YoungModulus, 0.0f, 0.0f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);
        Text("Poisson's ratio");
        InputDouble("##Poisson's ratio", &Material.PoissonRatio, 0.0f, 0.0f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);
        Text("Density (kg/m^3)");
        InputDouble("##Density", &Material.Density, 0.0f, 0.0f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);

        if (Button("Generate")) {
            const std::vector<int> excitable_vertex_indices{0, 1}; // todo
            const bool freq_control = true;
            ModalData->FaustDsp->SetCode(GenerateDsp(GenerateTets(ModalData->Mesh), Material, excitable_vertex_indices, freq_control));
        }
        if (ModalData->FaustDsp->Ui) {
            ModalData->FaustDsp->Ui->Draw();
        }
    }
}
