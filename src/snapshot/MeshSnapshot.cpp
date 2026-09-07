#include "mesh/Mesh.h"
#include "mesh/MeshBvh.h"
#include "mesh/MeshComponents.h"
#include "mesh/PrimitiveType.h"
#include "mesh/TetBuffers.h"
#include "render/MaterialComponents.h"
#include "render/MeshBuffers.h"
#include "scene/Entity.h"
#include "selection/SelectionComponents.h"
#include "snapshot/SnapshotRegistration.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportEvents.h"

namespace snapshot::detail {
template<> inline constexpr bool ForceFieldwise<PrimitiveShape> = true;

void RegisterMesh(Tables &tables) {
    Persistent<
        MeshActiveElement, MeshHandle, VertexStoreId, MeshElementSelection, MeshMaterialAssignment,
        MeshMaterialSlotSelection, PrimitiveShape, ShadeSmoothAngle, TetBuffers>(tables);
    Derived<MeshPositionsChanged, MeshBuffers, MeshShadingSummary, MeshBvh>(tables);
}
} // namespace snapshot::detail
