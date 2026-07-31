#pragma once

#include "SurfaceNoise.h"

#include <entt/entity/fwd.hpp>

#include <memory>

// The visible bumpiness a contact rides over, sampled along a path across a mesh's normal map.
// Its Spacing and Rms are mesh-local lengths, since a normal map's only size comes from the coordinates it is bound to.
// A contact multiplies both by the world scale of the node it is on, so one track serves every node instancing the mesh.
struct SurfaceRelief {
    std::shared_ptr<const RoughnessTrack> Track; // Shared with the pool slot a contact adopts it into.
    uint64_t Key{0}; // Content hash of the map, texel size, and scale, so one pool slot serves each distinct relief.
    uint64_t SourceKey{0}; // Hash of the map alone, distinguishing a surface edit from a relief edit without measuring the mesh.
};

// Rebuild a mesh entity's SurfaceRelief from its normal map, taken from its ContactSurface or else from its material.
// Removed when neither supplies a usable one.
// Pass `geometry_changed` when the mesh itself moved, since the track's texel size is measured from its texture coordinates.
void UpdateSurfaceRelief(entt::registry &, entt::entity mesh_entity, bool geometry_changed);
