#include "graph/Graph.h"
#include "graph/MaterialGraphs.h"
#include "graph/NodeTheme.h"
#include "mesh/MeshComponents.h"
#include "mesh/TetBuffers.h"
#include "render/MaterialComponents.h"
#include "scene/Entity.h"
#include "selection/SelectionComponents.h"
#include "snapshot/SnapshotRegistration.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportEvents.h"

namespace snapshot::detail {

void RegisterMesh(Tables &tables) {
    // Restore graph outputs, the result states and Boolean placements of the node occurrences they cooked and owned entities directly, and rederive only plans.
    Classify(
        tables,
        Concat(
            TypeList<
                MeshActiveElement, MeshHandle, EvaluatedMesh, VertexStoreId, MeshElementSelection, MeshMaterialAssignment,
                MeshMaterialSlotSelection, ShadeSmoothAngle, TetBuffers,
                graph::GraphNode, graph::NodeInputs, graph::GeometryGraph, graph::GraphNodes, graph::GraphActive, graph::NodeBypassed, graph::GraphLayout, graph::SubgraphItems, graph::SubgraphInterface, graph::Subgraph, graph::NodeError, graph::NodeErrors, graph::SourceState, graph::NodeResult, graph::NodeResults, graph::NodeExtrusion, graph::NodeExtrusions, graph::NodeHandle, graph::NodeHandles, graph::OperationHandle, graph::SelectionEditing, graph::BooleanPlacement, graph::BooleanPlacements, graph::GraphInstances, graph::GraphOwnedMesh, graph::NodeTheme, graph::InlineSelection, graph::PublishSelection, graph::OpValueBlock, graph::MaterialGraph>{},
            graph::NodeKinds{}
        ),
        TypeList<>{},
        TypeList<MeshGeometryDirty, MeshPositionsChanged, AuthoredCornerNormals, graph::CookPlan, graph::TimeReads, graph::TexcoordWriters, graph::StagedNode, graph::ItemNode, graph::MaterialProgram>{}
    );
}
} // namespace snapshot::detail
