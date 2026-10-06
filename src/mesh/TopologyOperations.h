#pragma once
#include "mesh/GeometrySelection.h"
#include "mesh/MeshTopology.h"
#include <optional>

struct Mesh;
struct MeshStore;
enum class GeometrySeparateMode : uint8_t { Selected,
                                            LooseParts,
                                            Material };
enum class GeometryMergeMode : uint8_t { Center,
                                         First,
                                         Last,
                                         Collapse,
                                         ByDistance };

// Plans read canonical geometry and use explicit selections; they never read editor state.
std::vector<MeshTopologyTask> SeparateGeometryTasks(const MeshStore &, const Mesh &, const GeometrySelection &, GeometrySeparateMode);
std::optional<MeshTopologyTask> BridgeEdgeLoopsTask(const MeshStore &, const Mesh &, const GeometrySelection &);
std::optional<MeshTopologyTask> GridFillTask(const MeshStore &, const Mesh &, const GeometrySelection &, uint32_t span);
std::optional<MeshTopologyTask> FillHolesTask(const MeshStore &, const Mesh &, uint32_t sides);
std::optional<MeshTopologyTask> ConvexHullTask(const MeshStore &, const Mesh &, const GeometrySelection &);
std::optional<MeshTopologyTask> RotateEdgesTask(const Mesh &, const GeometrySelection &);
std::optional<MeshTopologyTask> FillTask(const MeshStore &, const Mesh &, const GeometrySelection &);
MeshTopologyTask LoopCutTask(const Mesh &, uint32_t edge, uint32_t cuts);
MeshTopologyTask ExtrudeStepsTask(const Mesh &, const GeometrySelection &, Element, uint32_t steps, const mat3 &, vec3 translation, vec3 center);
// Plane operations affect the whole mesh. Execute each returned stage in order
// with ExecuteGeometryTopologyStages so later stages include newly cut faces.
std::vector<MeshTopologyTask> BisectTasks(const Mesh &, vec3 point, vec3 normal, bool clear_inner, bool clear_outer);
std::vector<MeshTopologyTask> SymmetrizeTasks(const Mesh &, uint8_t axis, bool negative);
MeshTopologyTask KnifeTask(const Mesh &, const GeometrySelection &, const mat4 &mesh_to_clip, vec2 extent, vec2 start, vec2 end);
// Reference chooses the survivor and is authoritative for First/Last and Center.
std::optional<MeshTopologyTask> MergeTask(const MeshStore &, const Mesh &, const GeometrySelection &, GeometryMergeMode, float distance, uint32_t reference);
