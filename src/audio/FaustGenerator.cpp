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
std::string GenerateDsp(const ModalSoundObject &model, const std::vector<uint32_t> &excitable_vertices, std::optional<float> fundamental_freq_opt, bool freq_control) {
    static constexpr std::string_view ModelName{"modalModel"};
    const auto model_dsp = m2f::modal2faust({model.ModeFreqs, model.ModeT60s, model.ModeGains}, {std::string(ModelName), freq_control});
    // Static code sections.
    static const auto gate = std::format("gate = button(\"{}[tooltip: When excitation source is 'Hammer', excites the vertex. With any excitation source, applies the current parameters.]\")", GateParamName);
    static constexpr std::string_view ToSAH{" : ba.sAndH(gate)"}; // Add a sample and hold on the gate, in serial, and end the expression.
    static constexpr std::string_view gain{"gain = hslider(\"Gain[scale:log]\",0.2,0,0.5,0.01)"};
    static constexpr std::string_view t60_scale{"t60Scale = hslider(\"t60[scale:log][tooltip: Scale T60 decay values of all modes by the same amount.]\",1,0.1,10,0.01)"};
    static constexpr std::string_view hammer_hardness{"hammerHardness = hslider(\"Hammer hardness[tooltip: Only has an effect when excitation source is 'Hammer'.]\",0.9,0,1,0.01)"};
    static constexpr std::string_view hammer_size{"hammerSize = hslider(\"Hammer size[tooltip: Only has an effect when excitation source is 'Hammer'.]\",0.1,0,1,0.01)"};
    static constexpr std::string_view hammer{"hammer(trig,hardness,size) = en.ar(att,att,trig)*no.noise : fi.lowpass(3,ctoff)\nwith{ ctoff = (1-size)*9500+500; att = (1-hardness)*0.01+0.001; }"};

    // Variable code sections.
    const float fundamental_freq = fundamental_freq_opt ?
        *fundamental_freq_opt :
        !model.ModeFreqs.empty() ? model.ModeFreqs.front() :
                                   440.0f;
    const uint32_t num_excite = excitable_vertices.size();
    const auto freq = std::format("freq = hslider(\"Frequency[scale:log][tooltip: Fundamental frequency of the model]\",{},60,26000,1){}", fundamental_freq, ToSAH);
    const auto ex_pos = std::format("exPos = nentry(\"{}\",{},0,{},1){}", ExciteIndexParamName, (num_excite - 1) / 2, num_excite - 1, ToSAH);
    const auto process = std::format("process = hammer(gate,hammerHardness,hammerSize) : {}({}exPos,t60Scale)*gain", ModelName, freq_control ? "freq," : "");
    return std::format("{}\n{};\n{}{};\n{}{};\n{};\n{};\n{};\n{}{};\n{};\n\n{};\n", model_dsp, gate, hammer_hardness, ToSAH, hammer_size, ToSAH, gain, freq, ex_pos, t60_scale, ToSAH, hammer, process);
}
} // namespace

void FaustGenerator::OnCreateModalSoundObject(entt::registry &r, entt::entity e) {
    const auto &model = r.get<ModalSoundObject>(e);
    auto dsp = GenerateDsp(model, model.ExcitableVertices, model.FundamentalFreq, true);
    OnCodeChanged(dsp);
}
void FaustGenerator::OnDestroyModalSoundObject(entt::registry &, entt::entity) {
    OnCodeChanged("");
}
