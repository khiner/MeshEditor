#pragma once

using Sample = float;
#ifndef FAUSTFLOAT
#define FAUSTFLOAT Sample
#endif

#include "CreateSvgResource.h"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

class CTree;
using Box = CTree *;
class dsp;
class FaustParams;

struct llvm_dsp_factory;

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
    void DrawGraph(const fs::path &svg_dir);

    Sample Get(std::string_view param_label) const;
    void Set(std::string_view param_label, Sample value) const;

    Sample *GetZone(std::string_view param_label) const;

    std::unique_ptr<SvgResource> FaustSvg;
    CreateSvgResource CreateSvg;

private:
    void Init();
    void Uninit();
    void Update();
    void DestroyDsp();

    void SaveSvg(const fs::path &dir);

    Box Box{nullptr};
    dsp *Dsp{nullptr};
    llvm_dsp_factory *DspFactory{nullptr};
    std::unique_ptr<FaustParams> Params;

    std::string Code;
    std::string ErrorMessage;
};
