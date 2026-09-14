#pragma once
#include "state/Entity.h"
void RegisterMeshStoreHandlers(state::Scene &);
namespace mtl {
struct BufferContext;
}
void InitMeshStore(state::Scene &, mtl::BufferContext &);
void ClearMeshStoreHandles(state::Scene &);
void DeinitMeshStore(state::Scene &);
