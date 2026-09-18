#pragma once

#include "gpu/MeshTopologyOp.h"
#include "gpu/Types.h"
#include "state/Entity.h"

#include <span>
#include <vector>

struct MeshTopologyTask {
    uint32_t SourceId;
    MeshTopologyOp Op;
    float Param0{}, Param1{};
    uint32_t Flags{};
    uint32_t Steps{1};
    // The source vertex a merge keeps and its output position.
    uint32_t TargetVertex{InvalidOffset};
    vec3 TargetPosition{};
    // The transform applied to copies, and the plane a plane-mode operator cuts or deletes by.
    mat3 CopyRotation{};
    vec3 CopyTranslation{};
    vec3 PlaneNormal{};
    float PlaneOffset{};
    // A knife: the mesh-to-clip transform, the target extent in pixels, and the segment in those pixels.
    mat4 ScreenTransform{};
    vec2 Extent{}, KnifeStart{}, KnifeEnd{};
    // A face list: a vertex count then positions, then a face count then each face's length and vertex indices.
    // Indices past the source vertex count name the listed vertices.
    // A cut list: a count then edge and parameter pairs.
    // A selection list: a count then element indices.
    std::vector<uint32_t> List{};
};

// Runs each task's operator on the GPU into a new store record.
// Returns each task's output id, or InvalidStoreId for a source without faces.
// The selection carries onto the surviving elements, and the caller derives the other domains and the summary.
std::vector<uint32_t> RunMeshTopology(state::Scene &, std::span<const MeshTopologyTask>);
