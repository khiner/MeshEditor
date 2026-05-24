#pragma once

#include <entt/entity/fwd.hpp>
#include <memory>
#include <optional>

struct PhysicsSimulationSettings;

// Goal: User edit of physics parameters updates Jolt reactively per frame via a handler doing minimal Jolt work.
struct PhysicsWorld {
    PhysicsWorld();
    ~PhysicsWorld();

    // Point the contact listener at the registry so it can resolve PhysicsMaterial
    // components by entity value from Body UserData. Call once after construction.
    void BindRegistry(const entt::registry &);

    // Build all physics bodies and constraints from ECS state from scratch.
    void Rebuild(entt::registry &);

    // Reactive change handlers — dispatch one ECS-component change to the cheapest
    // applicable Jolt path (cheap-apply when possible, body lifecycle when required).
    void OnShapeChange(entt::registry &, entt::entity);
    void OnMotionChange(entt::registry &, entt::entity);
    void OnMaterialChange(entt::registry &, entt::entity);
    void OnTriggerChange(entt::registry &, entt::entity);

    // Document-level resource change handlers. Update path re-applies targeted Jolt state
    // for bodies/constraints referencing the entity; destroy path clears dangling refs on
    // dependent components (which fires downstream per-entity handlers).
    void OnPhysicsMaterialDefChange(entt::registry &, entt::entity);
    void OnCollisionSystemDefChange(entt::registry &, entt::entity);
    void OnCollisionFilterDefChange(entt::registry &, entt::entity);
    void OnPhysicsJointDefChange(entt::registry &, entt::entity);

    // Rebuild all constraints from the current PhysicsJoint view.
    // No-op when neither joints_changed nor any body lifecycle ran this tick.
    void FlushJoints(const entt::registry &, bool joints_changed);

    void RecomputeSceneScale(const entt::registry &);

    bool HasBodies() const;
    uint32_t BodyCount() const;

    // Per-frame pose cache. Bake steps and records; Restore replays without re-simulating.
    void EnsureCacheRange(uint32_t start_frame, uint32_t end_frame);
    void ApplySimulationSettings(const PhysicsSimulationSettings &);
    void BakeFrame(entt::registry &, const PhysicsSimulationSettings &, uint32_t, float dt);
    void RestoreFrame(entt::registry &, uint32_t);
    bool HasCachedFrame(uint32_t) const;
    std::optional<uint32_t> BakedThrough() const; // nullopt if nothing baked since last clear
    void InvalidateFromFrame(uint32_t);
    void ClearCache();

    // null filter = permissive (collides with all).
    bool DoFiltersCollide(entt::entity, entt::entity) const;

    // Directional single-side test: does `source`'s membership intersect `target`'s collide-mask?
    // Effective collision requires this in both directions. Use for UI asymmetry visualization.
    bool DoesFilterAllow(entt::entity source, entt::entity target) const;

private:
    struct Impl;
    std::unique_ptr<Impl> P;

    void AddBody(entt::registry &, entt::entity);
    void RemoveBody(entt::registry &, entt::entity);
    void ApplyMotion(const entt::registry &, entt::entity);
    void ApplyShape(const entt::registry &, entt::entity);
    void ApplyMaterial(const entt::registry &, entt::entity);
    void ApplyMassPropertiesFromShape(const entt::registry &, entt::entity);
    void BuildJoint(const entt::registry &, entt::entity);
};

namespace physics {
PhysicsWorld &Init(entt::registry &);
void Deinit(entt::registry &);
} // namespace physics
