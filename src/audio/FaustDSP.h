#pragma once

using Sample = float;
#ifndef FAUSTFLOAT
#define FAUSTFLOAT Sample
#endif

#include "render/CreateSvgResource.h"

#include <expected>

namespace fs = std::filesystem;

class CTree;
using Box = CTree *;
class dsp;
class FaustParams;

class llvm_dsp_factory;

// `FaustDSP` is a wrapper around a Faust DSP and Box.
// It has a Faust DSP code string, and updates its DSP and Box instances to reflect the current code.
struct FaustDSP {
    FaustDSP(CreateSvgResource);
    ~FaustDSP();

    void SetCode(std::string_view code, uint32_t sample_rate) {
        Code = code;
        SampleRate = sample_rate;
        Update();
    }
    std::string_view GetCode() const { return Code; }
    const std::string &GetError() const { return ErrorMessage; }

    void Compute(uint32_t n, const Sample **input, Sample **output) const;

    void DrawParams();
    void DrawGraph();

    Sample Get(std::string_view param_label) const;
    void Set(std::string_view param_label, Sample value) const;

    Sample *GetZone(std::string_view param_label) const;

private:
    std::expected<void, std::string> CreateDsp();
    std::expected<void, std::string> Init();
    void Uninit();
    void Update();
    void DestroyDsp();

    void SaveSvg();

    Box Box{nullptr};
    dsp *Dsp{nullptr};
    llvm_dsp_factory *DspFactory{nullptr};
    std::unique_ptr<FaustParams> Params;

    std::string Code, ErrorMessage;
    uint32_t SampleRate{48'000}; // Output rate this DSP was last compiled at.

    constexpr static std::string_view RootSvgPath{"process.svg"};
    fs::path SelectedSvgPath{RootSvgPath};
    std::unique_ptr<SvgResource> FaustSvg;
    CreateSvgResource CreateSvg;
};
