#include "mesh/MeshComponents.h"
#include "mesh/TetBuffers.h"
#include "procedure/NodeTheme.h"
#include "procedure/MaterialGraphs.h"
#include "procedure/Procedure.h"
#include "render/MaterialComponents.h"
#include "scene/Entity.h"
#include "selection/SelectionComponents.h"
#include "snapshot/SnapshotRegistration.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportEvents.h"

namespace snapshot::detail {

void RegisterMesh(Tables &tables) {
    // Restore procedure outputs, the result states and Boolean placements of the node occurrences they cooked and owned entities directly, and rederive only plans.
    Classify(
        tables,
        Concat(
            TypeList<
                MeshActiveElement, MeshHandle, EvaluatedMesh, VertexStoreId, MeshElementSelection, MeshMaterialAssignment,
                MeshMaterialSlotSelection, ShadeSmoothAngle, TetBuffers,
                procedure::ProcedureNode, procedure::NodeInputs, procedure::GeometryProcedure, procedure::NetworkNodes, procedure::NetworkActive, procedure::NodeBypassed, procedure::NetworkLayout, procedure::NetworkSubnet, procedure::SubnetInterface, procedure::SubnetDefinition, procedure::NodeError, procedure::NodeErrors, procedure::SourceState, procedure::NodeResult, procedure::NodeResults, procedure::NodeExtrusion, procedure::NodeExtrusions, procedure::NodeHandle, procedure::NodeHandles, procedure::OperationHandle, procedure::SelectionEditing, procedure::BooleanPlacement, procedure::BooleanPlacements, procedure::ProcedureInstances, procedure::ProcedureOwnedMesh, procedure::NodeTheme, procedure::InlineSelection, procedure::PublishSelection, procedure::OpValueBlock, procedure::MaterialGraph>{},
            procedure::NodeKinds{}
        ),
        TypeList<>{},
        TypeList<MeshGeometryDirty, MeshPositionsChanged, AuthoredCornerNormals, procedure::CookPlan, procedure::TimeReads, procedure::TexcoordWriters, procedure::StagedNode, procedure::NetworkNode, procedure::MaterialProgram>{}
    );
}
} // namespace snapshot::detail
