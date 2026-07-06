#include "FaustDSP.h"

#include "FaustParams.h"
#include "Paths.h"
#include "render/SvgResource.h"

#include "draw/drawschema.hh" // faust/compiler/draw/drawschema.hh
#include "faust/dsp/llvm-dsp.h"

namespace {
fs::path FaustSvgDir() { return Paths::Base() / "MeshEditor-svg"; }
} // namespace

FaustDSP::FaustDSP(CreateSvgResource create_svg) : CreateSvg(std::move(create_svg)) {}
FaustDSP::~FaustDSP() { Uninit(); }

void FaustDSP::Compute(uint32_t n, const Sample **input, Sample **output) const {
    if (Dsp) Dsp->compute(n, const_cast<Sample **>(input), output);
}

void FaustDSP::DrawParams() {
    if (Params) Params->Draw();
}
void FaustDSP::DrawGraph() {
    if (!Box) return;

    if (!fs::exists(FaustSvgDir())) SaveSvg();
    if (const auto faust_svg_path = FaustSvgDir() / SelectedSvgPath; fs::exists(faust_svg_path)) {
        if (!FaustSvg || FaustSvg->Path != faust_svg_path) CreateSvg(FaustSvg, faust_svg_path);
        if (auto clickedLinkOpt = FaustSvg->Draw()) SelectedSvgPath = std::move(*clickedLinkOpt);
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

void FaustDSP::SaveSvg() { drawSchema(Box, FaustSvgDir().c_str(), "svg"); }

std::expected<void, std::string> FaustDSP::CreateDsp() {
    static constexpr std::string AppName{"MeshEditor"};
    static const fs::path LibrariesPath{(Paths::Base() / "../lib/faust/libraries").lexically_normal()};
    std::vector<const char *> argv{"-I", LibrariesPath.c_str()};
    if (std::is_same_v<Sample, double>) argv.push_back("-double");
    const int argc = argv.size();

    static int num_inputs, num_outputs;
    std::string error;
    Box = DSPToBoxes(AppName, Code, argc, argv.data(), &num_inputs, &num_outputs, error);
    if (!Box || !error.empty()) return std::unexpected{error.empty() ? "`DSPToBoxes` did not produce a result." : error};

    static constexpr int optimize_level = -1;
    DspFactory = createDSPFactoryFromBoxes(AppName, Box, argc, argv.data(), "", error, optimize_level);
    if (!DspFactory || !error.empty()) return std::unexpected{error.empty() ? "Could not create the Faust DSP factory." : error};

    Dsp = DspFactory->createDSPInstance();
    if (!Dsp) return std::unexpected{"Created the Faust DSP factory, but could not create the DSP instance."};

    Dsp->init(SampleRate);
    Params = std::make_unique<FaustParams>();
    Dsp->buildUserInterface(Params.get());
    return {};
}

std::expected<void, std::string> FaustDSP::Init() {
    if (Code.empty()) return {};

    createLibContext();
    auto result = CreateDsp();
    if (!result) {
        DestroyDsp();
        Box = nullptr;
    }
    return result;
}
void FaustDSP::Uninit() {
    if (Dsp || DspFactory) DestroyDsp();
    if (Box) Box = nullptr;
    ErrorMessage = "";
    destroyLibContext();

    FaustSvg.reset();
    SelectedSvgPath = RootSvgPath;
    if (fs::exists(FaustSvgDir())) fs::remove_all(FaustSvgDir());
}

void FaustDSP::Update() {
    Uninit();
    auto result = Init();
    ErrorMessage = result ? "" : std::move(result).error();
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
