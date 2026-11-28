#pragma once

#include "gpu/MeshTopologyOp.h"
#include "gpu/NamedAttributeRef.h"
#include "mesh/GeometrySelection.h"
#include <functional>
#include <optional>

// A node output a topology task writes: the name of its Bool attribute on the task's output record, which gains it where it lacks it, and the attribute's domain.
struct TopologyOutput {
    std::string Name;
    AttributeDomain Domain;
};
// A task's node outputs by output index, each true at the output elements of the output's set, or none for an output it does not write.
using TopologyOutputs = std::array<std::optional<TopologyOutput>, MaxTopologyOutputs>;

struct MeshTopologyTask {
    uint32_t SourceId;
    MeshTopologyOp Op;
    // A boolean's second source, combined with SourceId into a new mesh.
    uint32_t SecondSourceId{InvalidOffset};
    // A simplify's SimplifyJob record, whose elected edges a Decimate collapses.
    SlotOffset Simplify{};
    float Param0{}, Param1{};
    // Float attributes of the source holding each parameter per element of the domain the operator reads it in, or none to read the parameter.
    NamedAttributeRef Param0Attribute{}, Param1Attribute{};
    uint32_t Flags{};
    uint32_t Steps{1};
    // The source vertex a merge keeps and its output position.
    uint32_t TargetVertex{InvalidOffset};
    vec3 TargetPosition{};
    // The transform applied to copies or to a boolean's second source, and the plane a plane-mode operator cuts by.
    mat3 CopyRotation{};
    vec3 CopyTranslation{};
    vec3 PlaneNormal{};
    float PlaneOffset{};
    // A knife: the mesh-to-clip transform, the target extent in pixels, and the segment in those pixels.
    mat4 ScreenTransform{};
    vec2 Extent{}, KnifeStart{}, KnifeEnd{};
    // A level set's lattice as grid points per axis, whose source vertices are the lattice points in order, grid points before body centers.
    std::array<uint32_t, 3> LatticeSize{};
    // The output vertex, corner and face counts a count pass measured, which bound the output in place of the operator's bounds.
    std::optional<uvec3> Outputs{};
    // A primitive list holds a header of vertex count, boundary length, span, attribute source, boundary handles and primitive count.
    // Loops of (vertex, source-edge halfedge) pairs follow, where a loop of two creates a loose edge and a longer loop creates a face.
    // An InvalidOffset halfedge creates a new edge.
    // Grid interior positions derive on the GPU from the boundary, and other lists append no vertices.
    // AppendedBase + i refers to the i-th listed vertex, and existing handles are below AppendedBase.
    uint32_t AppendedBase{};
    // An edge list: a count then pairs of an edge and its cut count or its junction vertex.
    // A count list follows its pairs with partition tables.
    // A selection list: a count then element handles.
    // FlipNormals may narrow the selected faces to this explicit face list.
    // A ReplaceFaces list holds the removed count and vertices, then the source face count and record offsets.
    // Each record holds its source face and polygon count, then per polygon a corner count and corner and edge pairs.
    std::vector<uint32_t> List{};
    // The canonical selection, where None reads all three domains and Vertex derives the selected edges and faces from the selected vertices.
    // Edge or Face preserves those domains.
    // An empty selection selects nothing, and TopologyFlagSelectAll selects every live element in its place.
    Element SelectionElement{Element::None};
    SelectionSource Selection{};
    std::optional<GeometryMasks> SelectionOutput{};
    // A domain whose store selection masks rewrite the other two domains' store masks across the edited blocks, or None.
    Element StoreSelectionSource{Element::None};
    TopologyOutputs NodeOutputs{};
    // A placing operator's emission leaves every vertex at its base and writes the placement attributes in place of moving the displaced vertices.
    bool Places{};
};

struct ExtrusionSource {
    uint32_t Output, Source;
    vec3 Normal;
    bool operator==(const ExtrusionSource &) const = default;
};

// Label rounds the first submit encodes before the host checks convergence, doubling on each later submit.
constexpr uint32_t LabelRounds{16};

struct MeshStore;
struct MeshTopologyPushConstants;
MeshTopologyPushConstants TopologyPushConstants(const MeshStore &);
// The vertex attributes a placing emission wrote on the record: each vertex's two per-unit directions, and whether the operator displaces it.
std::array<uint32_t, 3> PlacementAttributes(const MeshStore &, uint32_t id);
// The vertices the record's last placing emission displaced, ascending.
std::vector<uint32_t> PlacedVertices(const MeshStore &, uint32_t id);
// Points the task's emitted selection at the store's masks of its record, so its emitted elements leave selected the way the editor's operators leave them.
void PublishStoreSelection(const MeshStore &, MeshTopologyTask &);

namespace state {
struct Registry;
} // namespace state
namespace mtl {
struct ComputeChain;
} // namespace mtl
struct MeshTopologyEdit;
struct GeometryTopologyResult {
    uint32_t GeometryId{};
    bool Changed{};
};

// Executes the tasks on canonical geometry without viewport, objects, history or render publication.
// Every task of a batch reads the common input geometry, so the source mutations are independent.
// Each result holds the task's destination record and whether the task changed it, and untouched elements keep their canonical handles.
// `constructed` reads the constructed edits of each chunk of tasks before their publication, given the index of the chunk's first task.
// `finish` finishes the published edits of each chunk of tasks on its chain, given the index of the chunk's first task, and FinishAll does when it is empty.
std::vector<GeometryTopologyResult> ExecuteGeometryTopology(state::Registry &, std::span<const MeshTopologyTask>, const std::function<void(mtl::ComputeChain &, std::span<MeshTopologyEdit>, uint32_t first)> &finish = {}, const std::function<void(std::span<const MeshTopologyEdit>, uint32_t first)> &constructed = {});

// Executes dependent tasks in order, each reading the preceding task's completed geometry.
// Returns one result per stage, no-op stages included, and later stages may retire earlier handles.
std::vector<GeometryTopologyResult> ExecuteGeometryTopologyStages(state::Registry &, std::span<const MeshTopologyTask>);

// Executes each boolean task into a new mesh, then collapses the zero-length edges that crossings at coincident points leave.
// Returns each task's new mesh, or none when the task changed nothing.
std::vector<std::optional<uint32_t>> ExecuteGeometryBooleans(state::Registry &, std::span<const MeshTopologyTask>);
