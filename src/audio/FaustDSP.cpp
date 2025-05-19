#include "FaustDSP.h"

#include "FaustParams.h"
#include "SvgResource.h"
#include "Worker.h"
#include "mesh/Mesh.h"

#include "draw/drawschema.hh" // faust/compiler/draw/drawschema.hh
#include "faust/dsp/llvm-dsp.h"

#include <vector>

namespace {
constexpr uint SampleRate = 48'000; // todo respect device sample rate
const fs::path FaustSvgDir{"MeshEditor-svg"};
} // namespace

FaustDSP::FaustDSP(CreateSvgResource create_svg) : CreateSvg(std::move(create_svg)) {}
FaustDSP::~FaustDSP() {
    Uninit();
}

void FaustDSP::Compute(uint n, const Sample **input, Sample **output) const {
    if (Dsp) Dsp->compute(n, const_cast<Sample **>(input), output);
}

void FaustDSP::DrawParams() {
    if (Params) Params->Draw();
}
void FaustDSP::DrawGraph() {
    if (!fs::exists(FaustSvgDir)) SaveSvg();

    if (const auto faust_svg_path = FaustSvgDir / SelectedSvgPath; fs::exists(faust_svg_path)) {
        if (!FaustSvg || FaustSvg->Path != faust_svg_path) {
            CreateSvg(FaustSvg, faust_svg_path);
        }
        if (auto clickedLinkOpt = FaustSvg->Render()) {
            SelectedSvgPath = std::move(*clickedLinkOpt);
        }
    }
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

void FaustDSP::SaveSvg() { drawSchema(Box, FaustSvgDir.c_str(), "svg"); }

void FaustDSP::Init() {
    if (Code.empty()) return;

    createLibContext();

    static constexpr std::string AppName{"MeshEditor"};
    static const std::string LibrariesPath{std::filesystem::relative("../lib/faust/libraries")};
    std::vector<const char *> argv{"-I", LibrariesPath.c_str()};
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

    FaustSvg.reset();
    SelectedSvgPath = RootSvgPath;
    if (fs::exists(FaustSvgDir)) fs::remove_all(FaustSvgDir);
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
