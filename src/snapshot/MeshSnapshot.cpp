#include "mesh/MeshComponents.h"
#include "mesh/TetBuffers.h"
#include "procedure/InlineSelection.h"
#include "procedure/NodeTheme.h"
#include "procedure/Procedure.h"
#include "render/MaterialComponents.h"
#include "scene/Entity.h"
#include "selection/SelectionComponents.h"
#include "snapshot/SnapshotRegistration.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportEvents.h"

namespace snapshot::detail {

void RegisterMesh(Tables &tables) {
    // Restore procedure outputs, the pins of the nodes they cooked and owned entities directly, and rederive only plans and node errors.
    Classify(
        tables,
        Concat(
            TypeList<
                MeshActiveElement, MeshHandle, EvaluatedMesh, VertexStoreId, MeshElementSelection, MeshMaterialAssignment,
                MeshMaterialSlotSelection, ShadeSmoothAngle, TetBuffers,
                procedure::ProcedureNode, procedure::NodeInputs, procedure::GeometryProcedure, procedure::SourceState, procedure::EditedInput, procedure::ProcedureInstances, procedure::ProcedureOwnedMesh, procedure::NodeLayout, procedure::NodeTheme, procedure::NodeSelected, procedure::InlineSelection, procedure::PublishSelection>{},
            procedure::NodeKinds{}
        ),
        TypeList<>{},
        TypeList<MeshGeometryDirty, MeshPositionsChanged, AuthoredCornerNormals, procedure::CookPlan, procedure::NodeError, procedure::StagedNode>{}
    );
}
} // namespace snapshot::detail
