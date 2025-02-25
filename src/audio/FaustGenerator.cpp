#include "FaustGenerator.h"

#include "ModalSoundObject.h"

#include "mesh2faust.h"
#include <entt/entity/registry.hpp>

#include <format>
#include <string>

FaustGenerator::FaustGenerator(entt::registry &r, OnFaustCodeChanged code_chanted) : R(r), OnCodeChanged(code_chanted) {
    R.on_construct<ModalSoundObject>().connect<&FaustGenerator::OnCreateModalSoundObject>(*this);
    R.on_destroy<ModalSoundObject>().connect<&FaustGenerator::OnDestroyModalSoundObject>(*this);
}
FaustGenerator::~FaustGenerator() = default;

namespace {
constexpr std::string_view ToSAH{" : ba.sAndH(gate)"}; // add a sample and hold on the gate in serial
FaustGenerator::ModalDsp GenerateModalDsp(const std::string_view model_name, const ModalSoundObject &obj, const std::vector<uint32_t> &excitable_vertices, std::optional<float> fundamental_freq_opt, bool freq_control) {
    static constexpr std::string_view ModelGain{"gain = hslider(\"Gain[scale:log]\",0.2,0,0.5,0.01)"};
    static constexpr std::string_view ModelT60Scale{"t60Scale = hslider(\"t60[scale:log][tooltip: Scale T60 decay values of all modes by the same amount.]\",1,0.1,10,0.01)"};
    const float fundamental_freq = fundamental_freq_opt ? *fundamental_freq_opt : !obj.ModeFreqs.empty() ? obj.ModeFreqs.front() :
                                                                                                           440.0f;
    const auto model_freq = std::format("freq = hslider(\"Frequency[scale:log][tooltip: Fundamental frequency of the model]\",{},60,26000,1){}", fundamental_freq, ToSAH);
    const auto num_excite = excitable_vertices.size();
    const auto model_ex_pos = std::format("exPos = nentry(\"{}\",{},0,{},1){}", ExciteIndexParamName, (num_excite - 1) / 2, num_excite - 1, ToSAH);

    const auto model = m2f::modal2faust({obj.ModeFreqs, obj.ModeT60s, obj.ModeGains}, {std::string(model_name), freq_control});
    const auto model_eval = std::format("{}({}exPos,t60Scale)*gain", model_name, freq_control ? "freq," : "");
    const auto model_definition = std::format("{}\n{};\n{};\n{};\n{}{};", model, ModelGain, model_freq, model_ex_pos, ModelT60Scale, ToSAH);
    return {model_definition, model_eval};
}
std::string GenerateDsp(FaustGenerator::ModalDsp modal_dsp) {
    static const auto HammerGate = std::format("gate = button(\"{}[tooltip: Applies the current parameters and excites the vertex.]\")", GateParamName);
    static constexpr std::string_view HammerHardness{"hammerHardness = hslider(\"Hammer hardness[tooltip: Only has an effect when excitation source is 'Hammer'.]\",0.9,0,1,0.01)"};
    static constexpr std::string_view HammerSize{"hammerSize = hslider(\"Hammer size[tooltip: Only has an effect when excitation source is 'Hammer'.]\",0.1,0,1,0.01)"};
    static constexpr std::string_view Hammer{"hammer(trig,hardness,size) = en.ar(att,att,trig)*no.noise : fi.lowpass(3,ctoff)\nwith{ ctoff = (1-size)*9500+500; att = (1-hardness)*0.01+0.001; }"};
    static constexpr std::string_view HammerEval = "hammer(gate,hammerHardness,hammerSize)";
    static const auto HammerDefinition = std::format("{};\n{}{};\n{}{};{};", HammerGate, HammerHardness, ToSAH, HammerSize, ToSAH, Hammer);

    return std::format("{}\n\n{}\n\nprocess = {} : {};\n", HammerDefinition, modal_dsp.Definition, HammerEval, modal_dsp.Eval);
}
} // namespace

void FaustGenerator::OnCreateModalSoundObject(entt::registry &r, entt::entity e) {
    const auto &model = r.get<ModalSoundObject>(e);
    auto dsp = GenerateDsp(GenerateModalDsp("modalModel", model, model.ExcitableVertices, model.FundamentalFreq, true));
    OnCodeChanged(dsp);
}
void FaustGenerator::OnDestroyModalSoundObject(entt::registry &, entt::entity) {
    OnCodeChanged("");
}
