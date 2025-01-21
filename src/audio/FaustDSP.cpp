#include "FaustDSP.h"

#include "FaustParams.h"
#include "Tets.h"
#include "Worker.h"
#include "mesh/Mesh.h"

#include "draw/drawschema.hh" // faust/compiler/draw/drawschema.hh
#include "faust/dsp/llvm-dsp.h"
#include "mesh2faust.h"
#include "tetMesh.h" // Vega

#include "tetgen.h" // Must be after any Faust includes, since it defined a `REAL` macro.

#include <filesystem>
#include <ranges>

using std::ranges::to;
using std::views::transform;

constexpr uint SampleRate = 48'000; // todo respect device sample rate

FaustDSP::FaustDSP() {}
FaustDSP::~FaustDSP() {
    Uninit();
}

void FaustDSP::Compute(uint n, const Sample **input, Sample **output) const {
    if (Dsp) Dsp->compute(n, const_cast<Sample **>(input), output);
}

void FaustDSP::DrawParams() {
    if (Params) Params->Draw();
}

Sample FaustDSP::Get(std::string_view param_label) const {
    if (auto *zone = GetZone(param_label)) return *zone;
    return 0;
}

void FaustDSP::Set(std::string_view param_label, Sample value) const {
    if (auto *zone = GetZone(param_label); zone) *zone = value;
}

Sample *FaustDSP::GetZone(std::string_view param_label) const {
    return Params ? Params->getZoneForLabel(param_label.data()) : nullptr;
}

void FaustDSP::SaveSvg() {
    drawSchema(Box, "MeshEditor-svg", "svg");
}

void FaustDSP::Init() {
    if (Code.empty()) return;

    createLibContext();

    static constexpr std::string AppName{"MeshEditor"};
    static const std::string LibrariesPath{std::filesystem::relative("../lib/faust/libraries")};
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
void FaustDSP::Uninit() {
    if (Dsp || DspFactory) DestroyDsp();
    if (Box) Box = nullptr;
    ErrorMessage = "";
    destroyLibContext();
}

void FaustDSP::Update() {
    Uninit();
    Init();
}

void FaustDSP::DestroyDsp() {
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

Mesh2FaustResult GenerateDsp(const tetgenio &tets, const AcousticMaterialProperties &material, const std::vector<uint> &excitable_vertices, bool freq_control, std::optional<float> fundamental_freq_opt) {
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
    const std::string_view model_dsp = m2f_result.modelDsp;
    if (model_dsp.empty()) return {"process = 0;", {}, {}, {}, {{}}, {}};

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
        .ModelDsp = std::format("{}{}", model_dsp, instrument.str()),
        .ModeFreqs = std::move(mode_freqs),
        .ModeT60s = std::move(m2f_result.model.modeT60s),
        .ModeGains = std::move(m2f_result.model.modeGains),
        .ExcitableVertices = std::move(excitable_vertices),
        .Material = std::move(material)
    };
}

void FaustDSP::GenerateDsp(
    const Mesh &mesh, vec3 mesh_scale, const std::vector<uint32_t> &excitable_vertices,
    std::optional<float> fundamental_freq, const AcousticMaterialProperties &material_props, bool quality_tets
) {
    DspGenerator = std::make_unique<Worker<Mesh2FaustResult>>("Generating modal audio model...", [&] {
        // todo Add an invisible tet mesh to the scene and support toggling between surface/volumetric tet mesh views.
        // scene.AddMesh(tets->CreateMesh(), {.Name = "Tet Mesh",  R.get<Model>(selected_entity).Transform;, .Select = false, .Visible = false});

        // We rely on `PreserveSurface` behavior for excitable vertices;
        // Vertex indices on the surface mesh must match vertex indices on the tet mesh.
        // todo display tet mesh in UI and select vertices for debugging (just like other meshes but restrict to edge view)
        while (!DspGenerator) {}
        DspGenerator->SetMessage("Generating tetrahedral mesh...");
        auto tets = GenerateTets(mesh, mesh_scale, {.PreserveSurface = true, .Quality = quality_tets});

        DspGenerator->SetMessage("Generating DSP...");
        return ::GenerateDsp(*tets, material_props, excitable_vertices, true, fundamental_freq);
    });
}
