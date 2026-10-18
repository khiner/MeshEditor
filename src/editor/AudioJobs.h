#pragma once
#include "Job.h"
#include "audio/AudioSystem.h"
#include "audio/ModalModelFile.h"
#include "audio/ModalWarmStart.h"
#include "editor/EditorSchema.h"
#include <FastFEM/SolveMonitor.h>

struct ModalGenerationResult {
    std::optional<ModalModelData> Model;
    ModalWarmStart WarmStart;
};

// An in-flight modal solve, at most one per sound entity.
struct ModalSolveJob {
    state::Entity Entity, Viewport;
    uint64_t Epoch;
    Job<ModalGenerationResult, fastfem::SolveMonitor> Work;
};
struct ModalSolveJobs {
    std::vector<std::shared_ptr<ModalSolveJob>> Jobs;
};
// Cancels every solve. Each result is discarded on arrival.
inline void Reset(ModalSolveJobs &jobs, state::Registry &) {
    for (const auto &job : jobs.Jobs) job->Work.RequestCancel();
}

struct SolveInputs {
    std::vector<vec3> Positions; // Mesh positions at the node's world scale (SI meters)
    std::vector<uint32_t> TriangleIndices;
    std::vector<uint32_t> Vertices; // Excitation vertices
    fastfem::SurfaceSolveConfig Config;
    fastfem::Discretization Discretization;
    vec3 NodeScale;
    size_t OperatorHash, ModalConfigHash;
};

bool IsSolving(const state::Registry &, state::Entity);
void LaunchModalSolve(state::Registry &, state::Entity viewport, state::Entity, const ModalSolveSettings &, const AcousticMaterial &);
SolveInputs BuildSolveInputs(const state::Registry &, state::Entity, state::Entity mesh_entity, const ModalSolveSettings &);
bool ModalModelStale(const state::Registry &, state::Entity, const SolveInputs &, const AcousticMaterial &);
