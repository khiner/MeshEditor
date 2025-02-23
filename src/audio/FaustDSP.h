#pragma once

using Sample = float;
#ifndef FAUSTFLOAT
#define FAUSTFLOAT Sample
#endif

#include "AcousticMaterial.h"
#include "CreateSvgResource.h"
#include "numeric/vec3.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

class CTreeBase;
using Box = CTreeBase *;
class dsp;
class FaustParams;

struct llvm_dsp_factory;

struct Mesh;
template<typename Result> struct Worker;

constexpr std::string_view ExciteIndexParamName{"Excite index"};
constexpr std::string_view GateParamName{"Gate"};

// `FaustDSP` is a wrapper around a Faust DSP and Box.
// It has a Faust DSP code string, and updates its DSP and Box instances to reflect the current code.
struct FaustDSP {
    FaustDSP(CreateSvgResource);
    ~FaustDSP();

    void SetCode(std::string_view code) {
        Code = code;
        Update();
    }
    std::string_view GetCode() const { return Code; }

    void Compute(uint32_t n, const Sample **input, Sample **output) const;

    void DrawParams();
    void DrawGraph();

    Sample Get(std::string_view param_label) const;
    void Set(std::string_view param_label, Sample value) const;

    Sample *GetZone(std::string_view param_label) const;

    void SaveSvg();

    std::unique_ptr<SvgResource> FaustSvg;
    CreateSvgResource CreateSvg;

private:
    void Init();
    void Uninit();
    void Update();
    void DestroyDsp();

    Box Box{nullptr};
    dsp *Dsp{nullptr};
    std::unique_ptr<FaustParams> Params;

    std::string ErrorMessage{""};

    std::string Code{""};
    llvm_dsp_factory *DspFactory{nullptr};
};
