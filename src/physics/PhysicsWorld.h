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

    // Build/teardown physics bodies from ECS state.
    void Rebuild(entt::registry &);
    void AddBody(entt::registry &, entt::entity);
    void RemoveBody(entt::registry &, entt::entity);

    // Step simulation, write results back to ECS Transform components.
    void Step(entt::registry &, float dt);

    bool HasBodies() const;
    uint32_t BodyCount() const;

    // Snapshot/restore for play/reset.
    void SaveSnapshot(entt::registry &);
    void RestoreSnapshot(entt::registry &);

    // Document-level resources (from glTF or UI).
    std::vector<PhysicsMaterial> Materials;
    std::vector<CollisionFilter> Filters;
    std::vector<PhysicsJointDef> JointDefs;

private:
    struct Impl;
    std::unique_ptr<Impl> P;
};
