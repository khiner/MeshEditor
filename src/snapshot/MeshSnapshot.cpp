#include "mesh/MeshComponents.h"
#include "mesh/TetBuffers.h"
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
    // Restore procedure outputs and owned entities directly, and rederive only plans and node errors.
    [&]<typename... Ks>(TypeList<Ks...>) {
        Classify(
            tables,
            TypeList<
                MeshActiveElement, MeshHandle, EvaluatedMesh, VertexStoreId, MeshElementSelection, MeshMaterialAssignment,
                MeshMaterialSlotSelection, PrimitiveShape, ShadeSmoothAngle, TetBuffers, ResyncState,
                procedure::GeometryProcedure, procedure::ProcedureInstances, procedure::ProcedureOwnedMesh, procedure::NodeLayout, procedure::NodeTheme, procedure::NodeSelected, procedure::InlineSelection, procedure::PublishSelection, Ks...>{},
            TypeList<MeshGeometryDirty, MeshPositionsChanged, AuthoredCornerNormals, procedure::CookPlan, procedure::NodeError>{}
        );
    }(procedure::NodeKinds{});
}
} // namespace snapshot::detail
