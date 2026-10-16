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
    Persistent<
        MeshActiveElement, MeshHandle, EvaluatedMesh, VertexStoreId, MeshElementSelection, MeshMaterialAssignment,
        MeshMaterialSlotSelection, PrimitiveShape, ShadeSmoothAngle, TetBuffers, ResyncState>(tables);
    Derived<MeshGeometryDirty, MeshPositionsChanged, AuthoredCornerNormals>(tables);
    // Restore procedure outputs and owned entities directly; rederive only plans and node errors.
    Persistent<procedure::GeometryProcedure, procedure::ProcedureInstances, procedure::ProcedureOwnedMesh, procedure::NodeLayout, procedure::NodeTheme, procedure::NodeSelected, procedure::InlineSelection, procedure::PublishSelection>(tables);
    [&]<typename... Ks>(action::TypeList<Ks...>) { Persistent<Ks...>(tables); }(procedure::NodeKinds{});
    Derived<procedure::CookPlan, procedure::NodeError>(tables);
}
} // namespace snapshot::detail
