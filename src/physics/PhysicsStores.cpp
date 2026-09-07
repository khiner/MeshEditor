#include "physics/PhysicsStores.h"
#include "physics/PhysicsTypes.h"
#include <entt/entity/registry.hpp>
namespace {
template<typename C>
void EmplaceIfAbsent(entt::registry &r, entt::entity e) {
    if (!r.all_of<C>(e)) r.emplace<C>(e);
}

} // namespace
void RegisterPhysicsStoreHandlers(entt::registry &r) {
    r.on_construct<PhysicsMotion>().connect<&EmplaceIfAbsent<PhysicsVelocity>>();
    r.on_destroy<PhysicsMotion>().connect<&entt::registry::remove<PhysicsVelocity>>();
    r.on_construct<ColliderShape>().connect<&EmplaceIfAbsent<ColliderMaterial>>();
    r.on_destroy<ColliderShape>().connect<&entt::registry::remove<ColliderMaterial>>();
}
