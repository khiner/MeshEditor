#pragma once

#include "gpu/Element.h"
#include "gpu/MeshTopologyOp.h"
#include "gpu/Types.h"
#include "mesh/GeometrySelection.h"
#include <span>
#include <vector>

struct MeshTopologyTask {
    uint32_t SourceId;
    MeshTopologyOp Op;
    float Param0{}, Param1{};
    uint32_t Flags{};
    uint32_t Steps{1};
    // The canonical source vertex a merge keeps and its output position.
    uint32_t TargetVertex{InvalidOffset};
    vec3 TargetPosition{};
    // The transform applied to copies, and the plane a plane-mode operator cuts or deletes by.
    mat3 CopyRotation{};
    vec3 CopyTranslation{};
    vec3 PlaneNormal{};
    float PlaneOffset{};
    // A knife: the mesh-to-clip transform, the target extent in pixels, and the segment in those pixels.
    mat4 ScreenTransform{};
    vec2 Extent{}, KnifeStart{}, KnifeEnd{};
    // A primitive list: new vertex count, grid boundary length and span, attribute-source handle,
    // boundary handles, then primitive count and loops of (vertex, source-edge halfedge) pairs.
    // Grid interior positions derive on the GPU from the boundary; other lists append no vertices.
    // Length two creates a loose edge; larger loops create faces. InvalidOffset means a new edge.
    // AppendedBase + i names the i-th listed vertex.
    // Existing handles are below AppendedBase.
    uint32_t AppendedBase{};
    // A cut list: a count then edge and parameter pairs.
    // A selection list: a count then canonical element handles.
    // FlipNormals may narrow the selected faces to this explicit face list.
    // ReplaceFaces: removed count and vertices, source face count and record offsets;
    // each record is (source face, polygon count, then corner count and corner/edge pairs per polygon).
    // Decimate: count then (vertex, surviving vertex, replacement x/y/z float bits).
    std::vector<uint32_t> List{};
    // Explicit canonical masks. None reads all three masks directly; Vertex derives
    // selected edges/faces from selected vertices. Edge/Face preserve those domains.
    // Empty masks select nothing; SelectAll deliberately selects all canonical elements.
    Element SelectionElement{Element::None};
    GeometrySelection Selection{};
};

struct MeshStore;
struct MeshTopologyPushConstants;
// Canonical arena bindings shared by topology emission and transfer passes.
MeshTopologyPushConstants TopologyPushConstants(const MeshStore &);
// A bound on the operator scratch words the task's edit lays out, as if its local source were its whole mesh.
// A mesh past the scratch budget counts as one at the budget's scale, since it takes a chunk to itself either way.
uint32_t TopologyScratchBound(const MeshStore &, const MeshTopologyTask &);

namespace state {
struct Scene;
}
struct GeometryTopologyResult {
    uint32_t GeometryId{};
    bool Changed{};
    GeometrySelection Created, Retained;
};

// Execute canonical geometry without viewport, objects, history, or render publication.
// A batch reads the common input geometry: source mutations must be independent.
// KeepSelectedFaces copies may precede their source's one mutation (separation).
// Dependent planner stages, such as BisectTasks/SymmetrizeTasks, use ExecuteGeometryTopologyStages.
// Results describe emitted elements; untouched elements keep their canonical handles.
std::vector<GeometryTopologyResult> ExecuteGeometryTopology(state::Scene &, std::span<const MeshTopologyTask>);

// Execute dependent tasks in order; each task reads the preceding task's completed geometry.
// Returns one result per stage, including no-op stages; later stages may retire earlier handles.
std::vector<GeometryTopologyResult> ExecuteGeometryTopologyStages(state::Scene &, std::span<const MeshTopologyTask>);
