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
        MeshMaterialSlotSelection, PrimitiveShape, ShadeSmoothAngle, TetBuffers>(tables);
    Derived<MeshGeometryDirty, MeshPositionsChanged, AuthoredCornerNormals, ResyncState>(tables);
    // A procedure's output record restores with the store, so only its plan, node errors and resync state derive.
    Persistent<procedure::GeometryProcedure, procedure::NodeLayout, procedure::InlineSelection>(tables);
    [&]<typename... Ks>(action::TypeList<Ks...>) { Persistent<Ks...>(tables); }(procedure::NodeKinds{});
    Derived<procedure::CookPlan, procedure::NodeError>(tables);
}
} // namespace snapshot::detail
