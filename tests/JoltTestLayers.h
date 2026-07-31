#pragma once

// The collision layers a standalone Jolt test scene needs: everything collides with Moving, and NonMoving is the static world.

#include "Jolt/Jolt.h"

#include "Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h"
#include "Jolt/Physics/Collision/ObjectLayer.h"

namespace Layers {
constexpr JPH::ObjectLayer NonMoving{0}, Moving{1};
} // namespace Layers
namespace BroadPhaseLayers {
constexpr JPH::BroadPhaseLayer NonMoving{0}, Moving{1};
constexpr JPH::uint Count{2};
} // namespace BroadPhaseLayers

class BPLayerInterface final : public JPH::BroadPhaseLayerInterface {
public:
    JPH::uint GetNumBroadPhaseLayers() const override { return BroadPhaseLayers::Count; }
    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer layer) const override { return layer == Layers::NonMoving ? BroadPhaseLayers::NonMoving : BroadPhaseLayers::Moving; }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char *GetBroadPhaseLayerName(JPH::BroadPhaseLayer) const override { return "Layer"; }
#endif
};
class ObjectVsBPFilter final : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer layer, JPH::BroadPhaseLayer bp) const override { return layer == Layers::NonMoving ? bp == BroadPhaseLayers::Moving : true; }
};
class ObjectPairFilter final : public JPH::ObjectLayerPairFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer a, JPH::ObjectLayer b) const override { return a == Layers::Moving || b == Layers::Moving; }
};
