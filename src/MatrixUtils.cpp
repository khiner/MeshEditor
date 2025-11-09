#include "MatrixUtils.h"

#include "Registry.h"
#include "Scale.h"
#include "Transform.h"
#include "numeric/mat4.h"

#include <entt/entity/registry.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

Transform GetTransform(const entt::registry &r, entt::entity e) {
    return {r.get<Position>(e).Value, r.get<Rotation>(e).Value, r.all_of<Scale>(e) ? r.get<Scale>(e).Value : vec3{1}};
}

mat4 ToLocalMatrix(const Transform &t) {
    return glm::translate(I4, t.P) * glm::mat4_cast(glm::normalize(t.R)) * glm::scale(I4, t.S);
}

mat4 ComputeWorldMatrix(const entt::registry &r, entt::entity e) {
    const auto t = GetTransform(r, e);
    const auto local = ToLocalMatrix(t);

    // Check if entity has a parent
    if (const auto *scene_node = r.try_get<SceneNode>(e); scene_node && scene_node->Parent != entt::null) {
        if (const auto *parent_inv = r.try_get<ParentInverse>(e)) {
            // Recursively compute parent's world matrix
            const auto parent_world = ComputeWorldMatrix(r, scene_node->Parent);
            // W_child = W_parent × ParentInverse × L_child
            return parent_world * parent_inv->Value * local;
        }
    }

    // No parent, local == world
    return local;
}
