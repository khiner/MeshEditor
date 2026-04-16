#pragma once

#include "PhysicsTypes.h"

#include <entt/entity/fwd.hpp>
#include <memory>

struct PhysicsWorld {
    PhysicsWorld();
    ~PhysicsWorld();

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

    // Query effective collision between two filter indices (for UI visualization).
    bool DoFiltersCollide(uint32_t a, uint32_t b) const;

    void UpdateFilterTable();

    // Document-level resources (from glTF or UI).
    std::vector<PhysicsMaterial> Materials;
    std::vector<CollisionFilter> Filters;
    std::vector<PhysicsJointDef> JointDefs;

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
