#pragma once
#include "Job.h"
#include "audio/AcousticMaterial.h"
#include "audio/AudioSystem.h"
#include "audio/ModalSolve.h"
#include "audio/ModalWarmStart.h"
#include <FastFEM/SolveMonitor.h>

struct ModalGenerationResult {
    std::filesystem::path ModelPath; // Result file, relative to ModalModelsDir(). Empty when the solve failed or was cancelled
    ModalWarmStart WarmStart;
};

// An in-flight modal solve, at most one per sound entity.
struct ModalSolveJob {
    entt::entity Entity, Viewport;
    Job<ModalGenerationResult, fastfem::SolveMonitor> Work;
};
struct ModalSolveJobs {
    std::vector<std::shared_ptr<ModalSolveJob>> Jobs;
};

struct SolveInputs {
    std::vector<vec3> Positions; // Mesh positions at the node's world scale (SI meters)
    std::vector<uint32_t> TriangleIndices;
    std::vector<uint32_t> Vertices; // Excitation vertices
    fastfem::SurfaceSolveConfig Config;
    fastfem::Discretization Discretization;
    vec3 NodeScale;
    size_t OperatorHash, ModalConfigHash;
};

bool IsSolving(const entt::registry &, entt::entity);
void LaunchModalSolve(entt::registry &, entt::entity viewport, entt::entity, const ModalSolveSettings &, const AcousticMaterial &);
SolveInputs BuildSolveInputs(const entt::registry &, entt::entity, entt::entity mesh_entity, const ModalSolveSettings &);
bool ModalModelStale(const entt::registry &, entt::entity, const SolveInputs &, const AcousticMaterial &);
