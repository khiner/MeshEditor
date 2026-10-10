#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "mesh/PrimitiveType.h"
#include "mesh/TetBuffers.h"
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
    // A procedure's output record restores with the store and its resync state, and its published instances and owned mesh with their entities, so only its plan and node errors derive.
    Persistent<procedure::GeometryProcedure, procedure::ProcedureInstances, procedure::ProcedureOwnedMesh, procedure::NodeLayout, procedure::NodeSelected, procedure::InlineSelection, procedure::PublishSelection>(tables);
    [&]<typename... Ks>(action::TypeList<Ks...>) { Persistent<Ks...>(tables); }(procedure::NodeKinds{});
    Derived<procedure::CookPlan, procedure::NodeError>(tables);
}
} // namespace snapshot::detail
