#include "physics/PhysicsStores.h"
#include "physics/PhysicsTypes.h"
#include "state/Registry.h"
namespace {
template<typename C>
void EmplaceIfAbsent(state::Registry &r, state::Entity e) {
    if (r.Restoring) return;
    if (!r.all_of<C>(e)) r.emplace<C>(e);
}

template<typename C>
void RemoveOwned(state::Registry &r, state::Entity e) {
    if (!r.Restoring) r.remove<C>(e);
}

} // namespace
void RegisterPhysicsStoreHandlers(state::Registry &r) {
    r.on_construct<PhysicsMotion, &EmplaceIfAbsent<PhysicsVelocity>>();
    r.on_destroy<PhysicsMotion, &RemoveOwned<PhysicsVelocity>>();
    r.on_construct<ColliderShape, &EmplaceIfAbsent<ColliderMaterial>>();
    r.on_destroy<ColliderShape, &RemoveOwned<ColliderMaterial>>();
}
