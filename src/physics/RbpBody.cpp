#include "RbpBody.h"

#include "Hull.h"
#include "TransformMath.h"
#include "World.h"

#include <cmath>
#include <stdexcept>

namespace physics {
namespace {
void CheckNonnegative(float value) {
    if (!std::isfinite(value) || value < 0) throw std::invalid_argument("Physics mass and inertia must be finite and nonnegative.");
}

rbp::float4 Rotation(quat q) {
    auto value = ToRbp(q);
    const float length = simd::length(value);
    if (!std::isfinite(length) || length < 1e-8f) throw std::invalid_argument("Physics orientation must be a finite nonzero quaternion.");
    return value / length;
}
rbp::BodyDesc DescribeBody(RbpBody &result, const Transform &node, const PhysicsMotion *motion, const PhysicsVelocity *velocity, bool sensor) {
    const auto cooked = result.CookedFrame;
    result.Frame = cooked;
    rbp::AuthoredMass mass{0, {0, 0, 0}};
    if (motion) {
        const float requested = motion->Mass.value_or(DefaultMass);
        CheckNonnegative(requested);
        mass.Mass = motion->IsKinematic ? 0 : requested;
        // Pinned translation still needs finite rotational mass properties.
        const float inertia_mass = requested > 0 ? requested : DefaultMass;
        const auto natural = result.NaturalMass;
        rbp::float3 moments{inertia_mass, inertia_mass, inertia_mass};
        if (natural.InvMass > 0) moments = inertia_mass * natural.InvMass / natural.InvInertiaLocal;

        if (motion->CenterOfMass) {
            result.Frame.Position = ToRbp(*motion->CenterOfMass * node.S);
            if (!std::isfinite(simd::length(result.Frame.Position))) throw std::invalid_argument("Physics centre of mass must be finite.");
        }
        if (motion->InertiaDiagonal) {
            moments = ToRbp(*motion->InertiaDiagonal);
            for (int axis = 0; axis < 3; ++axis) CheckNonnegative(moments[axis]);
            result.Frame.Orientation = Rotation(motion->InertiaOrientation.value_or(quat{1, 0, 0, 0}));
        } else if (motion->CenterOfMass) {
            // Carry the geometry's tensor to the authored centre before choosing principal axes.
            const rbp::float3 offset = cooked.Position - result.Frame.Position;
            const rbp::float3 axes[]{rbp::Rotate(cooked.Orientation, {1, 0, 0}), rbp::Rotate(cooked.Orientation, {0, 1, 0}), rbp::Rotate(cooked.Orientation, {0, 0, 1})};
            double tensor[3][3]{};
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    for (int axis = 0; axis < 3; ++axis) tensor[row][col] += double(axes[axis][row]) * moments[axis] * axes[axis][col];
                    tensor[row][col] += inertia_mass * ((row == col ? double(simd::dot(offset, offset)) : 0) - double(offset[row]) * offset[col]);
                }
            }
            const auto diagonal = rbp::DiagonalizeSymmetric(tensor);
            moments = {float(diagonal.Values.x), float(diagonal.Values.y), float(diagonal.Values.z)};
            result.Frame.Orientation = diagonal.Orientation;
        }
        mass.Inertia = motion->IsKinematic ? rbp::float3{0, 0, 0} : moments;
    }
    const auto node_pose = rbp::At(ToRbp(node.P), Rotation(node.R));
    rbp::Velocity initial{};
    if (motion && velocity) {
        initial.Angular = rbp::Rotate(node_pose.Orientation, ToRbp(velocity->Angular));
        initial.Linear = rbp::Rotate(node_pose.Orientation, ToRbp(velocity->Linear));
    }
    return {
        .Pose = rbp::ComposePose(node_pose, result.Frame),
        .Velocity = initial,
        .Shape = result.Shape,
        .Mass = mass,
        .Surface = ToRbp(PhysicsMaterial{}),
        .GravityScale = motion ? motion->GravityFactor : 0,
        .LinearDamping = motion ? motion->LinearDamping : 0,
        .AngularDamping = motion ? motion->AngularDamping : 0,
        .Sensor = sensor,
    };
}
void StoreInitialState(rbp::World &world, RbpBody &body, const rbp::BodyDesc &desc) {
    body.InitialPose = desc.Pose;
    body.InitialVelocity = desc.Velocity;
    if (body.Shape != rbp::NoIndex) world.Shapes[body.Shape].Local = rbp::ComposePose(Inverse(body.Frame), body.CookedFrame);
}
} // namespace

RbpBody BuildRbpBody(rbp::World &world, std::span<const rbp::Index> colliders, const Transform &node, const PhysicsMotion *motion, const PhysicsVelocity *velocity, bool sensor, const RbpBody *previous) {
    RbpBody result;
    try {
        if (!colliders.empty()) {
            result.Shape = world.AddCompound(colliders, &result.CookedFrame);
            if (result.Shape == rbp::NoIndex) throw std::runtime_error("RBP could not allocate the body's colliders.");
            if (motion) result.NaturalMass = rbp::MassProperties(world.Shapes[result.Shape], 1, world.ShapeVertices.All(), world.Shapes.All(), world.CompoundChildren.All());
        }
        const auto desc = DescribeBody(result, node, motion, velocity, sensor);
        StoreInitialState(world, result, desc);
        if (previous) {
            result.Body = previous->Body;
            if (!world.SetBodyShape(result.Body, result.Shape, 1, desc.Mass)) throw std::runtime_error("RBP could not replace the body's colliders.");
            auto &mass = world.Masses[result.Body];
            mass.GravityScale = desc.GravityScale;
            mass.LinearDamping = desc.LinearDamping;
            mass.AngularDamping = desc.AngularDamping;
            if (previous->Shape != rbp::NoIndex) world.RemoveShape(previous->Shape);
        } else {
            result.Body = world.AddBody(desc);
            if (result.Body == rbp::NoIndex) throw std::runtime_error("RBP could not allocate the body.");
        }
        return result;
    } catch (...) {
        if (result.Shape != rbp::NoIndex) world.RemoveShape(result.Shape);
        throw;
    }
}

void UpdateRbpBody(rbp::World &world, RbpBody &body, const Transform &node, const PhysicsMotion *motion, const PhysicsVelocity *velocity) {
    auto updated = body;
    const auto desc = DescribeBody(updated, node, motion, velocity, world.Filters[body.Body].Sensor);
    const auto mass = *desc.Mass;
    world.Masses[body.Body] = {
        .InvInertiaLocal = {mass.Inertia.x > 0 ? 1 / mass.Inertia.x : 0, mass.Inertia.y > 0 ? 1 / mass.Inertia.y : 0, mass.Inertia.z > 0 ? 1 / mass.Inertia.z : 0},
        .InvMass = mass.Mass > 0 ? 1 / mass.Mass : 0,
        .GravityScale = desc.GravityScale,
        .LinearDamping = desc.LinearDamping,
        .AngularDamping = desc.AngularDamping,
    };
    body = updated;
    StoreInitialState(world, body, desc);
}

CachedPose RbpNodePose(rbp::Pose body_pose, rbp::Pose frame) {
    const auto pose = rbp::ComposePose(body_pose, Inverse(frame));
    return {FromRbp(pose.Position), FromRbp(pose.Orientation)};
}
} // namespace physics
