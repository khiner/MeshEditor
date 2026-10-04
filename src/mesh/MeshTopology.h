#pragma once

#include "gpu/MeshTopologyOp.h"
#include "gpu/Types.h"
#include <vector>

struct MeshTopologyTask {
    uint32_t SourceId;
    MeshTopologyOp Op;
    float Param0{}, Param1{};
    uint32_t Flags{};
    uint32_t Steps{1};
    // The canonical source vertex a merge keeps and its output position.
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
    // A face list: new vertex count and positions, an existing attribute-source handle, then a face count and each face's length and vertex handles.
    // AppendedBase + i names the i-th listed vertex.
    // Existing handles are below AppendedBase.
    uint32_t AppendedBase{};
    // A cut list: a count then edge and parameter pairs.
    // A selection list: a count then canonical element handles.
    std::vector<uint32_t> List{};
};

struct MeshStore;
struct MeshTopologyPushConstants;
// Canonical arena bindings shared by topology emission and transfer passes.
MeshTopologyPushConstants TopologyPushConstants(const MeshStore &);
// A bound on the operator scratch words the task's edit lays out, as if its local source were its whole mesh.
// A mesh past the scratch budget counts as one at the budget's scale, since it takes a chunk to itself either way.
uint32_t TopologyScratchBound(const MeshStore &, const MeshTopologyTask &);
