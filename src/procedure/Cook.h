#pragma once

#include "action/ActionSchema.h"
#include "mesh/GeometrySelection.h"
#include "procedure/ProcedureSchema.h"

#include <optional>

namespace state {
struct Registry;
enum class Entity : uint32_t;
} // namespace state

struct MeshTopologyTask;

// The number of procedure cooks the settle pass has run.
struct CookCount {
    uint64_t Value{};
};

void RegisterProcedureHandlers(state::Registry &);
// Rebuilds the plans of changed procedures, and with `cook` evaluates each one whose graph, node parameters or inline selections changed.
void SettleProcedures(state::Registry &, bool cook);
// Adds the entities whose world transforms the pass published to the moved entities the next cook pass reads.
// Call before the registry clears its changes.
void RecordWorldTransformChanges(state::Registry &);
// Records the hash of the mesh storage state a settle pass that cooked one mesh alone ended with.
void FinishCookPass(state::Registry &);
// Keeps mesh storage as it stands as the state of each Source node with a pending state, which then holds that state's hash.
// Call where a history step ends.
void FinishSourceStates(state::Registry &);
// Ends a history step for the mesh storage states the Source nodes and node results refer to.
// Call after a history step is recorded.
void EndMeshStates(state::Registry &);
// Records in the history every mesh storage state the Source nodes and node results refer to.
// Call before the history's logs are read or copied.
void RecordMeshStates(state::Registry &);
// Makes the Source node's state pending again where a mesh draws its record without edits, so the step end keeps a write to the record outside the procedure.
void ReopenSourceState(state::Registry &, state::Entity mesh_entity);
// Forgets which nodes a history step's preview steps appended and which kept states restore the whole mesh storage exactly.
// Call where a history step ends.
void EndCookStep(state::Registry &);

namespace procedure {
// The elements a topology node edits, every element when In is absent, and the domain its selection was linked or captured in.
struct TopologyInput {
    std::optional<SelectionSource> In;
    Element Domain{Element::Vertex};
};
// An optional task is absent, and a task list empty, where the cook faults the node.
MeshTopologyTask PlanTopology(state::Registry &, const action::mesh::Delete &, uint32_t id, const TopologyInput &);
MeshTopologyTask PlanTopology(state::Registry &, const action::mesh::Dissolve &, uint32_t id, const TopologyInput &);
std::optional<MeshTopologyTask> PlanTopology(state::Registry &, const action::mesh::Merge &, uint32_t id, const TopologyInput &);
std::vector<MeshTopologyTask> PlanTopology(state::Registry &, const action::mesh::ConvexHull &, uint32_t id, const TopologyInput &);
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
std::optional<MeshTopologyTask> PlanTopology(state::Registry &, const action::mesh::BeautifyFaces &, uint32_t id, const TopologyInput &);
} // namespace procedure
