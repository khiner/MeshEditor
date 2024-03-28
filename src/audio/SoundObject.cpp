#include "SoundObject.h"

#include <format>

#include "imgui.h"
#include "tetgen.h"

#include "mesh2faust.h"

#include "faust/dsp/llvm-dsp.h"
using Sample = float;
#ifndef FAUSTFLOAT
#define FAUSTFLOAT Sample
#endif

#include "Material.h"
#include "RealImpact.h"
#include "Worker.h"
#include "mesh/Mesh.h"

using std::string, std::string_view;

string GenerateModelInstrumentDsp(const string_view model_dsp, int num_excite_pos) {
    static const string freq = "freq = hslider(\"Frequency[scale:log][tooltip: Fundamental frequency of the model]\",220,60,8000,1) : ba.sAndH(gate);";
    static const string source = "source = vslider(\"Excitation source [style:radio {'Hammer':0;'Audio input':1 }]\",0,0,1,1);";

    const string exPos = std::format("exPos = nentry(\"exPos\",{},0,{},1) : ba.sAndH(gate);", (num_excite_pos - 1) / 2, num_excite_pos - 1);

    static const string
        t60Scale = "t60Scale = hslider(\"t60[scale:log][tooltip: Resonance duration (s) of the lowest mode.]\",16,0,50,0.01) : ba.sAndH(gate);",
        t60Decay = "t60Decay = hslider(\"t60 Decay[scale:log][tooltip: Decay of modes as a function of their frequency, in t60 units.\nAt 1, the t60 of the highest mode will be close to 0 seconds.]\",0.80,0,1,0.01) : ba.sAndH(gate);",
        t60Slope = "t60Slope = hslider(\"t60 Slope[scale:log][tooltip: Power of the function used to compute the decay of modes t60 in function of their frequency.\nAt 1, decay is linear. At 2, decay slope has degree 2, etc.]\",2.5,1,6,0.01) : ba.sAndH(gate);",
        hammerHardness = "hammerHardness = hslider(\"hammerHardness[tooltip: Only has an effect when excitation source is 'Hammer'.]\",0.9,0,1,0.01) : ba.sAndH(gate);",
        hammerSize = "hammerSize = hslider(\"hammerSize[tooltip: Only has an effect when excitation source is 'Hammer'.]\",0.3,0,1,0.01) : ba.sAndH(gate);",
        gain = "gain = hslider(\"gain[scale:log]\",0.1,0,0.5,0.01);",
        gate = "gate = button(\"gate[tooltip: When excitation source is 'Hammer', excites the vertex. With any excitation source, applies the current parameters.]\");";

    // DSP code in addition to the model, to be appended to make it playable.
    static const string instrument = R"(
hammer(trig,hardness,size) = en.ar(att,att,trig)*no.noise : fi.lowpass(3,ctoff)
with{
  ctoff = (1-size)*9500+500;
  att = (1-hardness)*0.01+0.001;
};

process = hammer(gate,hammerHardness,hammerSize),_ : select2(source) : modalModel(freq,exPos,t60Scale,t60Decay,t60Slope)*gain;
)";

    std::stringstream full_instrument;
    full_instrument << source << '\n'
                    << gate << '\n'
                    << hammerHardness << '\n'
                    << hammerSize << '\n'
                    << gain << '\n'
                    << freq << '\n'
                    << exPos << '\n'
                    << t60Scale << '\n'
                    << t60Decay << '\n'
                    << t60Slope << '\n'
                    << '\n'
                    << instrument;

    return model_dsp.data() + full_instrument.str();
}

// `FaustDSP` is a wrapper around a Faust DSP and Box.
// It has a Faust DSP code string, and updates its DSP and Box instances to reflect the current code.
struct FaustDSP {
    FaustDSP();
    ~FaustDSP();

    inline static const string FaustDspFileExtension = ".dsp";

    Box Box{nullptr};
    dsp *Dsp{nullptr};

    string ErrorMessage{""};

    void SetCode(string_view code) {
        Code = std::move(code);
        Update();
    }

private:
    string Code{""};
    llvm_dsp_factory *DspFactory{nullptr};

    void Init();
    void Uninit();
    void Update(); // Sets `Box`, `Dsp`, and `ErrorMessage` based on the current `Code`.

    void DestroyDsp();
};

FaustDSP::FaustDSP() {
    Init();
}

FaustDSP::~FaustDSP() {
    Uninit();
}

void FaustDSP::DestroyDsp() {
    if (Dsp) {
        delete Dsp;
        Dsp = nullptr;
    }
    if (DspFactory) {
        deleteDSPFactory(DspFactory);
        DspFactory = nullptr;
    }
}

void FaustDSP::Init() {
    if (Code.empty()) return;

    static const string libraries_path = fs::relative("../lib/faust/libraries");
    std::vector<const char *> argv = {"-I", libraries_path.c_str()};
    if (std::is_same_v<Sample, double>) argv.push_back("-double");
    const int argc = argv.size();

    static int num_inputs, num_outputs;
    Box = DSPToBoxes("FlowGrid", Code, argc, argv.data(), &num_inputs, &num_outputs, ErrorMessage);

    if (Box && ErrorMessage.empty()) {
        static const int optimize_level = -1;
        DspFactory = createDSPFactoryFromBoxes("FlowGrid", Box, argc, argv.data(), "", ErrorMessage, optimize_level);
        if (DspFactory) {
            if (ErrorMessage.empty()) {
                Dsp = DspFactory->createDSPInstance();
                if (!Dsp) ErrorMessage = "Successfully created Faust DSP factory, but could not create the Faust DSP instance.";
            } else {
                deleteDSPFactory(DspFactory);
                DspFactory = nullptr;
            }
        }
    } else if (!Box && ErrorMessage.empty()) {
        ErrorMessage = "`DSPToBoxes` returned no error but did not produce a result.";
    }
}

void FaustDSP::Uninit() {
    if (Dsp || Box) {
        if (Dsp) DestroyDsp();
        if (Box) Box = nullptr;
    }
    ErrorMessage = "";
}

void FaustDSP::Update() {
    Uninit();
    Init();
}

FaustDSP FaustDsp;

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

string GenerateDsp(const tetgenio &tets, const MaterialProperties &material, const std::vector<int> &excitable_vertex_indices) {
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

    m2f::CommonArguments args{
        "modalModel",
        true, // freq control activated
        20, // lowest mode freq
        10000, // highest mode freq
        40, // number of synthesized modes (default is 20)
        80, // number of modes to be computed for the finite element analysis (default is 100)
        excitable_vertex_indices, // specific excitation positions
        int(excitable_vertex_indices.size()), // number of excitation positions (default is max: -1)
    };
    return m2f::mesh2faust(&volumetric_mesh, args);
}

SoundObject::SoundObject(const RealImpact &real_impact, const RealImpactListenerPoint &listener_point)
    : ListenerPosition(listener_point.GetPosition()), RealImpactData(listener_point.LoadImpactSamples(real_impact)) {}

SoundObject::SoundObject(const ::Mesh &mesh, vec3 listener_position)
    : ListenerPosition(listener_position), ModalData(mesh) {
}
SoundObject::SoundObject(const RealImpact &real_impact, const RealImpactListenerPoint &listener_point, const Mesh &mesh)
    : ListenerPosition(listener_point.GetPosition()), RealImpactData(listener_point.LoadImpactSamples(real_impact)), ModalData(mesh) {}

SoundObject::~SoundObject() = default;

void SoundObject::ProduceAudio(DeviceData device, float *output, uint frame_count) {
    if (Model == SoundObjectModel::RealImpact && RealImpactData) {
        if (RealImpactData->CurrentVertexIndex >= RealImpactData->ImpactSamples.size()) return;

        const uint sample_rate = device.SampleRate; // todo - resample from 48kHz to device sample rate if necessary
        (void)sample_rate; // Unused

        const auto &impact_samples = RealImpactData->ImpactSamples[RealImpactData->CurrentVertexIndex];
        for (uint i = 0; i < frame_count; ++i) {
            output[i] += RealImpactData->CurrentFrame < impact_samples.size() ? impact_samples[RealImpactData->CurrentFrame++] : 0.0f;
        }
    } else if (Model == SoundObjectModel::Modal) {
        FaustDsp.Dsp->compute(frame_count, nullptr, &output);
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
                    RealImpactData->CurrentFrame = 0;
                }
            }
            EndCombo();
        }
    } else if (Model == SoundObjectModel::Modal && ModalData) {
        if (Button("Generate")) {
            const auto material = MaterialPresets.at("Bell");
            const std::vector<int> excitable_vertex_indices{}; // todo
            const string model_dsp = GenerateDsp(GenerateTets(ModalData->Mesh), material, excitable_vertex_indices);
            FaustDsp.SetCode(model_dsp.empty() ? "process = _;" : GenerateModelInstrumentDsp(model_dsp, excitable_vertex_indices.size()));
        }
    }
}
