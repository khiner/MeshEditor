#include "physics/PhysicsStores.h"
#include "physics/PhysicsTypes.h"
#include "project/Registry.h"
#include <entt/entity/registry.hpp>
namespace {
template<typename C>
void EmplaceIfAbsent(entt::registry &r, entt::entity e) {
    if (project::Restoring(r)) return;
    if (!r.all_of<C>(e)) project::Emplace<C>(r, e);
}

template<typename C>
void RemoveOwned(entt::registry &r, entt::entity e) {
    if (!project::Restoring(r)) project::Remove<C>(r, e);
}

} // namespace
void RegisterPhysicsStoreHandlers(entt::registry &r) {
    r.on_construct<PhysicsMotion>().connect<&EmplaceIfAbsent<PhysicsVelocity>>();
    r.on_destroy<PhysicsMotion>().connect<&RemoveOwned<PhysicsVelocity>>();
    r.on_construct<ColliderShape>().connect<&EmplaceIfAbsent<ColliderMaterial>>();
    r.on_destroy<ColliderShape>().connect<&RemoveOwned<ColliderMaterial>>();
}
