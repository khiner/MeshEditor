#pragma once

#include "action/ActionSchema.h"
#include "graph/GraphSchema.h"
#include "mesh/GeometrySelection.h"

#include <functional>
#include <optional>

namespace state {
struct Registry;
enum class Entity : uint32_t;
} // namespace state

namespace store {
struct Hash128;
} // namespace store

struct MeshTopologyTask;
enum class EventPass;
namespace MTL {
class Texture;
} // namespace MTL

// The number of graph cooks the settle pass has run, of the nodes they ran, and of the images the evaluations that keep images stored.
struct CookCount {
    uint64_t Value{}, Nodes{}, Images{};
};
// The meshes the pass cooked only because the timeline's frame or time changed, sorted.
struct TimeCooks {
    std::vector<state::Entity> Meshes;
};

void RegisterGraphHandlers(state::Registry &);
// Rebuilds the plans of changed graphs, and in a Frame or Settle pass evaluates each one whose graph, node parameters or inline selections changed, or that reads the timeline's frame or time where they changed.
// A Restore pass takes the restored graphs to hold the restored frame.
// Each call takes the changes made since the last call.
void SettleGeometryGraphs(state::Registry &, EventPass);
// The meshes of the graphs that read the timeline's frame or time, and of those reading their meshes, sorted.
std::vector<state::Entity> TimeDependentMeshes(const state::Registry &);
// Evaluates the graphs of the time-dependent meshes, and of the time-dependent meshes they read, at the frame without publishing.
// Passes each mesh's record, in the order given, to `read`, then restores mesh storage exactly.
void ReadAtFrame(state::Registry &, std::span<const state::Entity> meshes, int frame, const std::function<void(std::span<const uint32_t>)> &read);
// Adds the entities whose world transforms the pass published to the moved entities the next cook pass reads.
// Call before the registry clears its changes.
void RecordWorldTransformChanges(state::Registry &);
// Records the hash of the mesh storage state a settle pass that cooked one mesh alone ended with.
void FinishCookPass(state::Registry &);
// Keeps mesh storage as it stands as the state of each Source node with a pending state, which then holds that state's hash.
// Call where a history step ends.
void FinishSourceStates(state::Registry &);
// The mesh storage states the Source nodes and node results refer to, sorted.
std::vector<store::Hash128> DocumentStates(const state::Registry &);
// Records in the history each kept mesh storage state that `states` lists or that a component changed since the last record refers to, and releases each other state kept since.
// Call where a history step ends.
void RecordMeshStates(state::Registry &, std::span<const store::Hash128> states = {});
// Makes the Source node's state pending again where a mesh draws its record without edits, so the step end keeps a write to the record outside the graph.
void ReopenSourceState(state::Registry &, state::Entity mesh_entity);
// Cooks the node's graph again from the node, after a write to its literals in the op value store.
void InvalidateNode(state::Registry &, state::Entity node);
// Whether a cook since the last ClearCookDependence took a path whose result depends on the cooks before it.
bool CooksDependOnEarlierCooks(const state::Registry &);
void ClearCookDependence(state::Registry &);
// Forgets which nodes a history step's preview steps appended and which kept states restore the whole mesh storage exactly.
// Call where a history step ends.
void EndCookStep(state::Registry &);

namespace graph {
// The image of an occurrence of a Bake node or a filter, its opaque view for display, and its texels along each axis.
// Hold keeps both textures alive while a view that draws them holds it.
struct ImageTextures {
    MTL::Texture *Image, *Shown;
    uint32_t Width, Height;
    std::shared_ptr<const void> Hold;
};
// The image the mesh's graph holds for the occurrence at the cooks' frame, or none.
// Evaluates the nodes the occurrence reads, publishing nothing, when no evaluation since they last changed did.
std::optional<ImageTextures> OccurrenceImageOf(state::Registry &, state::Entity mesh, std::span<const state::Entity> path);

// The elements a topology node edits, every element when In is absent, and the domain its selection was linked or captured in.
struct TopologyInput {
    std::optional<SelectionSource> In;
    Element Domain{Element::Vertex};
};
// An optional task is absent, and a task list empty, where the cook faults the node.
MeshTopologyTask PlanTopology(state::Registry &, const action::mesh::Delete &, uint32_t id, const TopologyInput &);
MeshTopologyTask PlanTopology(state::Registry &, const action::mesh::Dissolve &, uint32_t id, const TopologyInput &);
std::optional<MeshTopologyTask> PlanTopology(state::Registry &, const action::mesh::Merge &, uint32_t id, const TopologyInput &);
MeshTopologyTask PlanTopology(state::Registry &, const action::mesh::Wireframe &, uint32_t id, const TopologyInput &);
MeshTopologyTask PlanTopology(state::Registry &, const action::mesh::Bevel &, uint32_t id, const TopologyInput &);
MeshTopologyTask PlanTopology(state::Registry &, const action::mesh::TrisToQuads &, uint32_t id, const TopologyInput &);
MeshTopologyTask PlanTopology(state::Registry &, const action::mesh::SplitFaces &, uint32_t id, const TopologyInput &);
MeshTopologyTask PlanTopology(state::Registry &, const action::mesh::FlipNormals &, uint32_t id, const TopologyInput &);
MeshTopologyTask PlanTopology(state::Registry &, const action::mesh::Split &, uint32_t id, const TopologyInput &);
MeshTopologyTask PlanTopology(state::Registry &, const action::mesh::Duplicate &, uint32_t id, const TopologyInput &);
std::optional<MeshTopologyTask> PlanTopology(state::Registry &, const Extrude &, uint32_t id, const TopologyInput &);
std::optional<MeshTopologyTask> PlanTopology(state::Registry &, const action::mesh::Subdivide &, uint32_t id, const TopologyInput &);
MeshTopologyTask PlanTopology(state::Registry &, const action::mesh::EdgeSplit &, uint32_t id, const TopologyInput &);
MeshTopologyTask PlanTopology(state::Registry &, const action::mesh::Poke &, uint32_t id, const TopologyInput &);
MeshTopologyTask PlanTopology(state::Registry &, const action::mesh::Inset &, uint32_t id, const TopologyInput &);
MeshTopologyTask PlanTopology(state::Registry &, const action::mesh::Solidify &, uint32_t id, const TopologyInput &);
MeshTopologyTask PlanTopology(state::Registry &, const action::mesh::ConnectVertices &, uint32_t id, const TopologyInput &);
} // namespace graph
