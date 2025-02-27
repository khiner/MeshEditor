#include "FaustGenerator.h"

#include "ModalSoundObject.h"
#include "Registry.h"

#include "mesh2faust.h"
#include <entt/entity/registry.hpp>

#include <format>
#include <ranges>
#include <sstream>
#include <string>

FaustGenerator::FaustGenerator(entt::registry &r, OnFaustCodeChanged code_chanted) : R(r), OnCodeChanged(code_chanted) {
    R.on_construct<ModalSoundObject>().connect<&FaustGenerator::OnCreateModalSoundObject>(*this);
    R.on_destroy<ModalSoundObject>().connect<&FaustGenerator::OnDestroyModalSoundObject>(*this);
}
FaustGenerator::~FaustGenerator() = default;

namespace {
using ModalDsp = FaustGenerator::ModalDsp;
constexpr std::string_view ToSAH{" : ba.sAndH(gate)"}; // add a sample and hold on the gate in serial
ModalDsp GenerateModalDsp(std::string_view model_name, const ModalSoundObject &obj, bool freq_control) {
    static constexpr std::string_view ModelGain{"gain = hslider(\"Gain[scale:log]\",0.2,0,0.5,0.01)"};
    static constexpr std::string_view ModelT60Scale{"t60Scale = hslider(\"t60[scale:log][tooltip: Scale T60 decay values of all modes by the same amount.]\",1,0.1,10,0.01)"};
    const float fundamental_freq = obj.FundamentalFreq ? *obj.FundamentalFreq : !obj.ModeFreqs.empty() ? obj.ModeFreqs.front() :
                                                                                                         440.0f;
    const auto model_freq = std::format("freq = hslider(\"Frequency[scale:log][tooltip: Fundamental frequency of the model]\",{},60,26000,1){}", fundamental_freq, ToSAH);
    const uint num_excitable = obj.Excitable.ExcitableVertices.size();
    const auto model_ex_pos = std::format("exPos = nentry(\"{}\",{},0,{},1){}", ExciteIndexParamName, (num_excitable - 1) / 2, num_excitable - 1, ToSAH);

    const auto model = m2f::modal2faust({obj.ModeFreqs, obj.ModeT60s, obj.ModeGains}, {"modalModel", freq_control});
    const auto model_definition = std::format("{} = environment {{\n{}\n{};\n{};\n{};\n{}{};\n}};", model_name, model, ModelGain, model_freq, model_ex_pos, ModelT60Scale, ToSAH);
    const auto model_eval = std::format("{}.modalModel({}{}.exPos,{}.t60Scale)*{}.gain", model_name, freq_control ? std::format("{}.freq,", model_name) : "", model_name, model_name, model_name);
    return {model_definition, model_eval};
}
std::string GenerateDsp(const std::unordered_map<entt::entity, ModalDsp> &modal_dsp_by_entity) {
    if (modal_dsp_by_entity.empty()) return "";

    static const auto HammerGate = std::format("gate = button(\"{}[tooltip: Applies the current parameters and excites the vertex.]\")", GateParamName);
    static constexpr std::string_view HammerHardness{"hammerHardness = hslider(\"Hammer hardness[tooltip: Only has an effect when excitation source is 'Hammer'.]\",0.9,0,1,0.01)"};
    static constexpr std::string_view HammerSize{"hammerSize = hslider(\"Hammer size[tooltip: Only has an effect when excitation source is 'Hammer'.]\",0.1,0,1,0.01)"};
    static constexpr std::string_view Hammer{"hammer(trig,hardness,size) = en.ar(att,att,trig)*no.noise : fi.lowpass(3,ctoff)\nwith{ ctoff = (1-size)*9500+500; att = (1-hardness)*0.01+0.001; }"};
    static constexpr std::string_view HammerEval = "hammer(gate,hammerHardness,hammerSize)";
    static const auto HammerDefinition = std::format("import(\"stdfaust.lib\");{};\n{}{};\n{}{};{};", HammerGate, HammerHardness, ToSAH, HammerSize, ToSAH, Hammer);

    const auto modal_dsps = modal_dsp_by_entity | std::views::values;
    std::stringstream modal_definitions;
    for (const auto &modal_dsp : modal_dsps) modal_definitions << modal_dsp.Definition << '\n';
    // clang doesn't have join_with yet
    std::stringstream modal_evals;
    size_t i = 0;
    for (const auto &modal_dsp : modal_dsps) {
        if (i > 0) modal_evals << ",";
        modal_evals << modal_dsp.Eval;
        ++i;
    }
    return std::format("{}\n\n{}\n\nprocess = {} <: ({});\n", HammerDefinition, modal_definitions.str(), HammerEval, modal_evals.str());
}
} // namespace

void FaustGenerator::OnCreateModalSoundObject(entt::registry &r, entt::entity e) {
    const auto &model = r.get<ModalSoundObject>(e);
    const auto name = GetName(r, e);
    ModalDspByEntity[e] = GenerateModalDsp(name, model, true);
    OnCodeChanged(GenerateDsp(ModalDspByEntity));
}
void FaustGenerator::OnDestroyModalSoundObject(entt::registry &, entt::entity e) {
    ModalDspByEntity.erase(e);
    OnCodeChanged(GenerateDsp(ModalDspByEntity));
}
