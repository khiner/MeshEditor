#pragma once
#include "mesh/MeshTopology.h"

#include <functional>

struct Mesh;
enum class PlaneSideKind : uint32_t;
namespace mtl {
struct ComputeChain;
}
enum class GeometrySeparateMode : uint8_t { Selected,
                                            LooseParts,
                                            Material };
enum class GeometryMergeMode : uint8_t { Center,
                                         First,
                                         Last,
                                         Collapse,
                                         ByDistance };

// A separation keeps each group of the mode in its own copy and then deletes those groups, leaving in place the group Blender leaves.
std::vector<MeshTopologyTask> SeparateGeometryTasks(state::Registry &, const Mesh &, const GeometrySelection &, GeometrySeparateMode);
// Bridge two selected chains of equal openness with a strip of faces, or fill one closed even chain with a grid.
// Records the reads on the chain, which gather with the other lanes' reads, and calls `then` with the task after the submits that complete them.
void RecordBridgeEdgeLoopsTask(state::Registry &, mtl::ComputeChain &, const Mesh &, const GeometrySelection &, std::function<void(std::optional<MeshTopologyTask>)> then);
void RecordGridFillTask(state::Registry &, mtl::ComputeChain &, const Mesh &, const GeometrySelection &, uint32_t span, std::function<void(std::optional<MeshTopologyTask>)> then);
// Cut each selected edge into pieces no longer than `length` as manifold's refine to length divides them, or none when no edge is long enough.
std::optional<MeshTopologyTask> SubdivideLengthTask(state::Registry &, const Mesh &, const GeometrySelection &, float length);
// Refine the closed mesh onto manifold's smooth patches through its derived corner normals, with `cuts` new vertices per edge.
std::vector<MeshTopologyTask> SmoothSurfaceTasks(const Mesh &, uint32_t cuts);
// The hull of the selected vertices in place of the selection, or none when fewer than four are selected or they are degenerate.
std::vector<MeshTopologyTask> ConvexHullTasks(state::Registry &, const Mesh &, const GeometrySelection &);
// Records the hull request in the record's lane of the chain, which joins the other pending lanes' requests in one quickhull once the chain submits, and calls `then` with the tasks.
void RecordConvexHullTasks(state::Registry &, mtl::ComputeChain &, const Mesh &, const GeometrySelection &, std::function<void(std::vector<MeshTopologyTask>)> then);
// Fill the selected boundary loops of at most `sides` edges, every loop when `sides` is zero, else the selected chains or the selected points as one face.
// The passes that read the mesh record in lanes of the chain, and `then` takes the task, or none, after the submits that complete them.
void RecordFillTask(state::Registry &, mtl::ComputeChain &, const Mesh &, const GeometrySelection &, uint32_t sides, std::function<void(std::optional<MeshTopologyTask>)> then);
MeshTopologyTask ExtrudeStepsTask(const Mesh &, const GeometrySelection &, Element, uint32_t steps, const mat3 &, vec3 translation, vec3 center);
// Cut every edge whose ends lie beyond the plane side tolerance on opposite sides at the crossing.
// Every vertex within the tolerance of the plane, cut or kept, snaps onto it.
MeshTopologyTask PlaneCutTask(const Mesh &, vec3 point, vec3 normal);
// Delete the vertices on the side of the plane with their edges and faces, or none when no vertex is there.
// The pass that reads the mesh records in a lane of the chain, and `then` takes the task after the submit that completes it.
void RecordPlaneSideDeleteTask(state::Registry &, mtl::ComputeChain &, const Mesh &, vec3 point, vec3 normal, PlaneSideKind, float tolerance, std::function<void(std::optional<MeshTopologyTask>)> then);
MeshTopologyTask KnifeTask(const Mesh &, const mat4 &mesh_to_clip, vec2 extent, vec2 start, vec2 end);
// The reference, a selected vertex, chooses the survivor and the position of First, Last and Center.
std::optional<MeshTopologyTask> MergeTask(state::Registry &, const Mesh &, const GeometrySelection &, GeometryMergeMode, float distance, uint32_t reference);
// Plan a boolean between two closed meshes, or none when either is open.
// The rotation and translation take the second into the first's space and preserve orientation, as the caller establishes.
std::optional<MeshTopologyTask> BooleanTask(const Mesh &first, const Mesh &second, BooleanOperation, const mat3 &rotation, vec3 translation);
