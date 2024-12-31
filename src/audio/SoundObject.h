#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "Variant.h"
#include "numeric/vec3.h"

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

// Actions
namespace SoundObjectAction {
struct SetModel {
    SoundObjectModel Model;
};
struct SelectVertex {
    uint Vertex;
};
struct Excite {
    uint Vertex;
    float Force;
};
struct SetExciteForce {
    float Force;
};
using Any = std::variant<SetModel, SelectVertex, Excite, SetExciteForce>;
} // namespace SoundObjectAction

struct AudioBuffer;
struct Tets;
struct AcousticMaterial;
struct Mesh2FaustResult;

struct ModalAudioModel;
struct ImpactAudioModel;

template<typename Result> struct Worker;

struct SvgResource;
using CreateSvgResource = std::function<void(std::unique_ptr<SvgResource> &, std::filesystem::path)>;

// Represents a rigid mesh object that generate an audio stream in response to an impact at a given vertex.
struct SoundObject {
    SoundObject(CreateSvgResource);
    ~SoundObject();

    std::vector<uint> ExcitableVertices;
    // The vertex currently selected for excitation.
    uint SelectedVertex{0};

    void Apply(SoundObjectAction::Any);

    void ProduceAudio(const AudioBuffer &) const;
    std::optional<SoundObjectAction::Any> RenderControls(std::string_view name, const Tets &, AcousticMaterial &);

    void SetImpactFrames(std::unordered_map<uint, std::vector<float>> &&impact_frames_by_vertex);

    std::optional<uint> FindNearestExcitableVertex(const Tets &, vec3 position) const;

private:
    void SetVertex(uint);
    void SetVertexForce(float);

    std::unique_ptr<ModalAudioModel> ModalModel;
    std::unique_ptr<ImpactAudioModel> ImpactModel;

    SoundObjectModel Model{SoundObjectModel::ImpactAudio};
    std::unique_ptr<Worker<Mesh2FaustResult>> DspGenerator;
    CreateSvgResource CreateSvg;
};
