#pragma once
#include "entt_fwd.h"
void RegisterMeshStoreHandlers(entt::registry &);
namespace mtl {
struct BufferContext;
}
void InitMeshStore(entt::registry &, mtl::BufferContext &);
void ClearMeshStoreHandles(entt::registry &);
void DeinitMeshStore(entt::registry &);
