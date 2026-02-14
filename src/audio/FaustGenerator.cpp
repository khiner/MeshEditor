#include "FaustGenerator.h"

#include "Entity.h"
#include "ModalSoundObject.h"

#include <entt/entity/registry.hpp>

#include <format>
#include <ranges>
#include <sstream>
#include <string>

FaustGenerator::FaustGenerator(entt::registry &r, OnFaustCodeChanged code_changed) : R(r), OnCodeChanged(code_changed) {
    R.on_construct<ModalSoundObject>().connect<&FaustGenerator::OnCreateModalSoundObject>(*this);
    R.on_destroy<ModalSoundObject>().connect<&FaustGenerator::OnDestroyModalSoundObject>(*this);
}
FaustGenerator::~FaustGenerator() = default;

namespace {
// Generate DSP code from modal modes.
std::string GenerateModalModelDsp(const ModalModes &modes, std::string_view model_name, bool freq_control) {
    const auto &freqs = modes.Freqs;
    const auto &gains = modes.Gains;
    const auto &t60s = modes.T60s;
    const auto n_modes = freqs.size();
    const auto n_ex_pos = gains.size();

    std::stringstream dsp;
    dsp << model_name << "(" << (freq_control ? "freq," : "")
        << "exPos,t60Scale) = _ <: "
           "par(mode,nModes,pm.modeFilter(modeFreqs(mode),modesT60s(mode),"
           "modesGains(int(exPos),mode))) :> /(nModes)\n"
        << "with{\n"
        << "nModes = " << n_modes << ";\n";
    if (n_ex_pos > 1) dsp << "nExPos = " << n_ex_pos << ";\n";

    dsp << "modeFreqsUnscaled(n) = ba.take(n+1,(";
    for (size_t mode = 0; mode < n_modes; ++mode) {
        dsp << freqs[mode];
        if (mode < n_modes - 1) dsp << ",";
    }
    dsp << "));\n";
    dsp << "modeFreqs(mode) = ";
    if (freq_control) dsp << "freq*modeFreqsUnscaled(mode)/modeFreqsUnscaled(0)";
    else dsp << "modeFreqsUnscaled(mode)";
    dsp << ";\n";

    dsp << "modesGains(p,n) = waveform{";
    for (size_t ex_pos = 0; ex_pos < gains.size(); ++ex_pos) {
        for (size_t mode = 0; mode < gains[ex_pos].size(); ++mode) {
            dsp << gains[ex_pos][mode];
            if (mode < n_modes - 1) dsp << ",";
        }
        if (ex_pos < gains.size() - 1) dsp << ",";
    }
    dsp << "},int(p*nModes+n) : rdtable" << (freq_control ? " : select2(modeFreqs(n)<(ma.SR/2-1),0)" : "") << ";\n";

    dsp << "modesT60s(n) = t60Scale * ba.take(n+1,(";
    for (size_t mode = 0; mode < n_modes; ++mode) {
        dsp << t60s[mode];
        if (mode < n_modes - 1) dsp << ",";
    }
    dsp << "));\n";
    dsp << "};\n";

    return dsp.str();
}

using ModalDsp = FaustGenerator::ModalDsp;
// Note: Frequency control is not physically accurate as it doesn't scale mode dampings.
ModalDsp GenerateModalDsp(std::string_view model_name, const ModalSoundObject &obj, bool freq_control) {
    const float fundamental_freq = obj.Modes.Freqs.front();
    static constexpr std::string_view ModelGain{"gain = hslider(\"Gain[scale:log]\",0.2,0,0.5,0.01)"};
    static constexpr std::string_view ModelT60Scale{"t60Scale = hslider(\"t60[scale:log][tooltip: Scale T60 decay values of all modes by the same amount.]\",1,0.1,10,0.01)"};
    const auto model_freq = std::format("freq = hslider(\"Frequency[scale:log][tooltip: Fundamental frequency of the model]\",{},50,16000,1)", fundamental_freq);
    const uint num_excitable = obj.Excitable.ExcitableVertices.size();
    const auto model_ex_pos = std::format("exPos = nentry(\"{}\",{},0,{},1)", ExciteIndexParamName, (num_excitable - 1) / 2, num_excitable - 1);
    const auto model = GenerateModalModelDsp(obj.Modes, "modalModel", freq_control);
    const auto model_definition = std::format("{} = environment {{\n{}\n{};\n{};\n{};\n{};\n}};", model_name, model, ModelGain, model_freq, model_ex_pos, ModelT60Scale);

    constexpr std::string_view ToSAH{" : ba.sAndH(gate)"}; // add a sample and hold on the gate in serial
    const auto freq = freq_control ? std::format("{}.freq{},", model_name, ToSAH) : "";
    const auto model_eval = std::format("{}.modalModel({}{}.exPos{},{}.t60Scale{})*{}.gain", model_name, freq, model_name, ToSAH, model_name, ToSAH, model_name);
    return {std::string(model_name), model_definition, model_eval};
}
std::string GenerateDsp(const std::unordered_map<entt::entity, ModalDsp> &modal_dsp_by_entity) {
    if (modal_dsp_by_entity.empty()) return "";

    static const auto HammerGate = std::format("gate = button(\"../../{}[tooltip: Applies the current parameters and excites the vertex.]\")", GateParamName);
    static constexpr std::string_view HammerHardness{"hammerHardness = hslider(\"Hammer hardness\",0.9,0,1,0.01)"};
    static constexpr std::string_view HammerSize{"hammerSize = hslider(\"Hammer size\",0.1,0,1,0.01)"};
    static constexpr std::string_view Hammer{"hammer(trig,hardness,size) = en.ar(att,att,trig)*no.noise : fi.lowpass(3,ctoff)\nwith{ ctoff = (1-size)*9500+500; att = (1-hardness)*0.01+0.001; }"};
    static constexpr std::string_view HammerEval = "hammer(gate,hammerHardness,hammerSize)";
    static const auto HammerDefinition = std::format("{};\n{};\n{};\n{};", HammerGate, HammerHardness, HammerSize, Hammer);

    const auto modal_dsps = modal_dsp_by_entity | std::views::values;
    std::stringstream modal_definitions;
    for (const auto &modal_dsp : modal_dsps) modal_definitions << modal_dsp.Definition << "\n\n";

    const bool multiple_models = modal_dsp_by_entity.size() > 1;
    // clang doesn't have join_with yet
    std::stringstream models_eval;
    if (multiple_models) models_eval << "tgroup(\"Models\", ";
    size_t i = 0;
    for (const auto &modal_dsp : modal_dsps) {
        if (i > 0) models_eval << ",";
        models_eval << std::format("vgroup(\"{}\", {})", modal_dsp.Name, modal_dsp.Eval);
        ++i;
    }
    if (multiple_models) models_eval << ")";

    const auto switch_definition = multiple_models ? std::format("N={};\nswitchN(n, s) = par(i, n , _*(i==s));\nmodelIndex = nentry(\"Excite model\", 0, 0, N-1, 1);", modal_dsp_by_entity.size()) : "";
    const std::string_view switch_eval = multiple_models ? "switchN(N, modelIndex) : " : "";
    const std::string_view mix = multiple_models ? "/(N)" : "_";
    auto hammer = std::format("vgroup(\"Hammer\", {})", HammerEval);
    // Since models are in a tab group when there are multiple, the gate param needs to be one level deeper for its path to match.
    if (multiple_models) hammer = std::format("vgroup(\"\", {})", hammer);
    return std::format(
        "import(\"stdfaust.lib\");\n\n{}\n\n{}\n\n{}\n\nprocess = {} <: {}{} :> {};\n",
        HammerDefinition, switch_definition, modal_definitions.str(), hammer, switch_eval, models_eval.str(), mix
    );
}
} // namespace

void FaustGenerator::OnCreateModalSoundObject(const entt::registry &r, entt::entity e) {
    const auto &model = r.get<const ModalSoundObject>(e);
    auto model_name = GetName(r, e);
    std::ranges::replace(model_name, ' ', '_'); // Make sure name is a valid identifier for Faust (todo handle invalid characters)
    ModalDspByEntity[e] = GenerateModalDsp(model_name, model, true);
    OnCodeChanged(GenerateDsp(ModalDspByEntity));
}
void FaustGenerator::OnDestroyModalSoundObject(const entt::registry &, entt::entity e) {
    ModalDspByEntity.erase(e);
    OnCodeChanged(GenerateDsp(ModalDspByEntity));
}
