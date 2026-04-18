#pragma once

#include "PhysicsTypes.h"

struct Mesh;

// Resolve the mesh-data entity for an instance or mesh entity.
entt::entity FindMeshEntity(const entt::registry &, entt::entity);

// Pick an initial ColliderShape from `entity`'s PrimitiveShape if any, else from the mesh BBox.
ColliderShape CreateInitialCollider(const entt::registry &, entt::entity);

// Rewrite shape dimensions to match the mesh BBox, preserving the shape kind.
// No-op for Plane, ConvexHull, TriangleMesh.
void FitShapeToMesh(PhysicsShape &, const Mesh &);

// Patch entity's ColliderShape to its mesh-BBox fit.
// Patches unconditionally so a changes::PhysicsShape event fires even for kinds with no fittable dims.
void RefitAutoFitShape(entt::registry &, entt::entity);
