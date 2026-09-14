#include "physics/PhysicsStores.h"
#include "physics/PhysicsTypes.h"
#include "state/Scene.h"
namespace {
template<typename C>
void EmplaceIfAbsent(state::Scene &r, state::Entity e) {
    if (r.Restoring) return;
    if (!r.all_of<C>(e)) r.emplace<C>(e);
}

template<typename C>
void RemoveOwned(state::Scene &r, state::Entity e) {
    if (!r.Restoring) r.remove<C>(e);
}

} // namespace
void RegisterPhysicsStoreHandlers(state::Scene &r) {
    r.on_construct<PhysicsMotion>().connect<&EmplaceIfAbsent<PhysicsVelocity>>();
    r.on_destroy<PhysicsMotion>().connect<&RemoveOwned<PhysicsVelocity>>();
    r.on_construct<ColliderShape>().connect<&EmplaceIfAbsent<ColliderMaterial>>();
    r.on_destroy<ColliderShape>().connect<&RemoveOwned<ColliderMaterial>>();
}
