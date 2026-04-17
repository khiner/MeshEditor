#pragma once

#include "PhysicsTypes.h"

#include <entt/entity/fwd.hpp>
#include <memory>

// Goal: Every user edit of a physics parameter updates Jolt reactively per frame,
// via a handler doing the minimum Jolt work the API allows.
// (We currently over-update in some cases for simplicity.)
struct PhysicsWorld {
    PhysicsWorld();
    ~PhysicsWorld();

    // Point the contact listener at the registry so it can resolve PhysicsMaterial
    // components by entity value from Body UserData. Call once after construction.
    void BindRegistry(const entt::registry &);

    // Simulation state
    float TimeStep{1.0f / 60.0f};
    vec3 Gravity{0.0f, -9.81f, 0.0f};
    int SubSteps{1};

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

    // Step simulation, write results back to ECS Transform components.
    void Step(entt::registry &, float dt);

    bool HasBodies() const;
    uint32_t BodyCount() const;

    // Snapshot/restore for play/reset.
    void SaveSnapshot(entt::registry &);
    void RestoreSnapshot(entt::registry &);

    // Query effective collision between two filter entities (for UI visualization).
    // null filter = permissive (collides with all).
    bool DoFiltersCollide(entt::entity a, entt::entity b) const;

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
