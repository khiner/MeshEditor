#pragma once

using Sample = float;
#ifndef FAUSTFLOAT
#define FAUSTFLOAT Sample
#endif

#include "AcousticMaterial.h"
#include "numeric/vec3.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

class tetgenio;

struct Mesh2FaustResult {
    std::string ModelDsp; // Faust DSP code defining the model function.
    std::vector<float> ModeFreqs; // Mode frequencies
    std::vector<float> ModeT60s; // Mode T60 decay times
    std::vector<std::vector<float>> ModeGains; // Mode gains by [exitation position][mode]
    std::vector<uint32_t> ExcitableVertices; // Excitable vertices
    AcousticMaterialProperties Material;
};

Mesh2FaustResult GenerateDsp(
    const tetgenio &, const AcousticMaterialProperties &,
    const std::vector<uint32_t> &excitable_vertices,
    bool freq_control = false, std::optional<float> fundamental_freq_opt = {}
);

class CTreeBase;
using Box = CTreeBase *;
class dsp;
class FaustParams;

struct llvm_dsp_factory;

struct Mesh;
template<typename Result> struct Worker;

constexpr std::string ExciteIndexParamName{"Excite index"};
constexpr std::string GateParamName{"Gate"};

// `FaustDSP` is a wrapper around a Faust DSP and Box.
// It has a Faust DSP code string, and updates its DSP and Box instances to reflect the current code.
struct FaustDSP {
    FaustDSP();
    ~FaustDSP();

    void SetCode(std::string_view code) {
        Code = code;
        Update();
    }
    std::string_view GetCode() const { return Code; }

    void Compute(uint32_t n, const Sample **input, Sample **output) const;

    void DrawParams();

    Sample Get(std::string_view param_label) const;
    void Set(std::string_view param_label, Sample value) const;

    Sample *GetZone(std::string_view param_label) const;

    void SaveSvg();

    void GenerateDsp(const Mesh &, vec3 mesh_scale, const std::vector<uint32_t> &excitable_vertices, std::optional<float> fundamental_freq, const AcousticMaterialProperties &, bool quality_tets);

    std::unique_ptr<Worker<Mesh2FaustResult>> DspGenerator;

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
