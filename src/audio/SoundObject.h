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

// All model-specific data.
struct ImpactAudioModel {
    ImpactAudioModel(std::unordered_map<uint, std::vector<float>> &&impact_frames_by_vertex, uint vertex);
    ~ImpactAudioModel();

    const ImpactAudioModel &operator=(ImpactAudioModel &&) noexcept;

    std::unordered_map<uint, std::vector<float>> ImpactFramesByVertex;
    uint MaxFrame;
    uint CurrentFrame{MaxFrame}; // Start at the end, so it doesn't immediately play.
    std::unique_ptr<Waveform> Waveform; // Current vertex's waveform

    void Start() { CurrentFrame = 0; }
    void Stop() { CurrentFrame = MaxFrame; }
    bool IsStarted() const { return CurrentFrame != MaxFrame; }

    bool CanStrike() const { return bool(Waveform); }
    void SetVertex(uint);
    void SetVertexForce(float force) {
        if (force > 0 && !IsStarted()) Start();
        else if (force == 0 && IsStarted()) Stop();
    }

    void Draw() const;
};

struct ModalAudioModel {
    ModalAudioModel(Mesh2FaustResult &&, uint vertex);
    ~ModalAudioModel();

    uint ModeCount() const { return ModeFreqs.size(); }

    void ProduceAudio(float *input, float *output, uint frame_count) const;

    bool CanStrike() const;
    void SetVertex(uint);
    void SetVertexForce(float);
    void Stop() { SetVertexForce(0); }

    void SetParam(std::string_view param_label, Sample param_value) const;

    void Draw(uint *selected_vertex_index); // Renders a vertex index dropdown.

    std::unique_ptr<Waveform> Waveform; // Recorded waveform

private:
    std::vector<uint> ExcitableVertices;

    // todo use Mesh2FaustResult
    std::unique_ptr<FaustDSP> FaustDsp;
    std::vector<float> ModeFreqs{};
    std::vector<float> ModeT60s{};
    std::vector<std::vector<float>> ModeGains{};

    std::unique_ptr<ImpactRecording> ImpactRecording;
    std::optional<size_t> HoveredModeIndex;
};

// Represents a rigid mesh object that generate an audio stream for a listener at a given position
// in response to an impact at a given vertex.
struct SoundObject : AudioSource {
    // All SoundObjects have a modal audio model.
    // If `impact_frames_by_vertex` is non-empty, the object also has an impact audio model.
    SoundObject(std::string_view name, const Tets &, const std::optional<std::string_view> &material_name, vec3 listener_position, uint listener_entity_id);
    ~SoundObject();

    const std::string Name;
    const Tets &Tets;
    std::string_view MaterialName;
    MaterialProperties Material;
    vec3 ListenerPosition;
    uint ListenerEntityId{0};
    std::vector<uint> ExcitableVertices;
    uint CurrentVertex{0}, CurrentVertexIndicatorEntityId{0};

    std::optional<ModalAudioModel> ModalModel{};
    std::optional<ImpactAudioModel> ImpactModel{};

    void ProduceAudio(DeviceData, float *input, float *output, uint frame_count) override;
    void RenderControls();

    void SetModel(SoundObjectModel);
    void SetImpactFrames(std::unordered_map<uint, std::vector<float>> &&impact_frames_by_vertex);

    void SetVertex(uint);
    void SetVertexForce(float);
    std::optional<uint> FindNearestExcitableVertex(vec3 position);

private:
    SoundObjectModel Model{SoundObjectModel::ImpactAudio};
    std::unique_ptr<Worker<Mesh2FaustResult>> DspGenerator;
};
