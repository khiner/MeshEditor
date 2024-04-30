#pragma once

#include <optional>
#include <unordered_map>
#include <vector>

#include "numeric/vec3.h"

#include "AudioSource.h"
#include "MaterialProperties.h"

using Sample = float;
#ifndef FAUSTFLOAT
#define FAUSTFLOAT Sample
#endif

enum class SoundObjectModel {
    // Play back recordings of impacts on the object at provided listener points/vertices.
    ImpactAudio,
    // Model a rigid body's response to an impact using modal analysis/synthesis:
    // - transforming the object's geometry into a tetrahedral volume mesh
    // - using FEM to estimate the mass/sg/damping matrices from the mesh
    // - estimating the dominant modal frequencies and amplitudes using an eigenvalues/eigenvector solver
    // - simulating the object's response to an impact by exciting the modes associated with the impacted vertex
    Modal,
};

struct FaustDSP;
struct Tets;
struct Waveform;
template<typename Result> struct Worker;

struct Mesh2FaustResult {
    std::string ModelDsp; // Faust DSP code defining the model function.
    std::vector<float> ModeFreqs; // Mode frequencies
    std::vector<float> ModeT60s; // Mode T60 decay times
    std::vector<std::vector<float>> ModeGains; // Mode gains by [exitation position][mode]
    std::vector<uint> ExcitableVertices; // Copy of the excitable vertices used for model generation.
};

struct ImpactRecording {
    static constexpr uint FrameCount = 208592; // Same length as RealImpact recordings.
    float Frames[FrameCount];
    uint CurrentFrame{0};
    bool Complete{false};
};

// All model-specific data needed to render audio.
namespace SoundObjectData {
struct ImpactAudio {
    ImpactAudio(std::unordered_map<uint, std::vector<float>> &&impact_frames_by_vertex);
    ~ImpactAudio();

    const ImpactAudio &operator=(ImpactAudio &&) noexcept;

    std::unordered_map<uint, std::vector<float>> ImpactFramesByVertex;
    uint CurrentFrame{0};
    std::unique_ptr<Waveform> Waveform; // Current vertex's waveform

    void SetVertex(uint);
};

struct Modal {
    Modal(Mesh2FaustResult &&);
    ~Modal();

    void ProduceAudio(float *input, float *output, uint frame_count) const;
    void Draw(uint *selected_vertex_index); // Renders a vertex index dropdown.

    void SetParam(std::string_view param_label, Sample param_value) const;

    std::vector<uint> ExcitableVertices;

private:
    // todo use Mesh2FaustResult
    std::unique_ptr<FaustDSP> FaustDsp;
    std::vector<float> ModeFreqs{};
    std::vector<float> ModeT60s{};
    std::vector<std::vector<float>> ModeGains{};

    std::unique_ptr<Waveform> Waveform; // Recorded waveform
    std::unique_ptr<ImpactRecording> ImpactRecording;
    std::optional<size_t> HoveredModeIndex;
};
} // namespace SoundObjectData

// Represents a rigid mesh object that generate an audio stream for a listener at a given position
// in response to an impact at a given vertex.
struct SoundObject : AudioSource {
    // All SoundObjects have a modal audio model. If `impact_frames_by_vertex` is non-empty, the object also has an impact audio model.
    SoundObject(const Tets &, const std::optional<std::string> &material_name, vec3 listener_position, uint listener_entity_id);
    ~SoundObject();

    const Tets &Tets;
    std::string MaterialName;
    MaterialProperties Material;
    vec3 ListenerPosition;
    uint ListenerEntityId{0};
    std::vector<uint> ExcitableVertices;
    uint CurrentVertex{0}, CurrentVertexIndicatorEntityId{0};

    std::optional<SoundObjectData::Modal> ModalModel{};
    std::optional<SoundObjectData::ImpactAudio> ImpactAudioModel{};

    void SetModel(SoundObjectModel);
    void ProduceAudio(DeviceData, float *input, float *output, uint frame_count) override;
    void RenderControls();
    void SetImpactFrames(std::unordered_map<uint, std::vector<float>> &&impact_frames_by_vertex);

private:
    SoundObjectModel Model{SoundObjectModel::ImpactAudio};
    std::unique_ptr<Worker<Mesh2FaustResult>> DspGenerator;
};
