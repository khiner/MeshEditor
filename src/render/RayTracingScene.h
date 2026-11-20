#pragma once
#include "metal/Buffer.h"
#include "metal/PassChain.h"
#include <memory>
#include <vector>

namespace state {
struct Registry;
}
struct GpuBuffers;
struct MeshStore;
struct MainPipeline;
struct SceneViewUBO;

struct RayTracingMaterials {
    bool Coverage{}, BlendedTransmission{}, Opaque{};
    // All casters outside these instances obey the same opaque triangle culling policy.
    std::vector<uint32_t> ComplexCasterSlots;
    bool CullOpaqueBackfaces{};
    bool OpaqueSingleSided{true}, OpaqueDoubleSided{true};
};
RayTracingMaterials RayTracingMaterialUsage(const MeshStore &, const GpuBuffers &, const SceneViewUBO &, bool classify_casters);

// Acceleration structures reference canonical position arenas; only indices and hit mappings are derived.
class RayTracingScene {
public:
    explicit RayTracingScene(const mtl::Context &);
    ~RayTracingScene();
    RayTracingScene(const RayTracingScene &) = delete;
    // Returns the hit instance buffer address for this sample, or zero for an empty scene.
    uint64_t Build(mtl::PassChain &, const MeshStore &, const GpuBuffers &, const SceneViewUBO &, const state::Registry &, const MainPipeline &);
    void Bind(mtl::Command &, uint64_t hit_instances) const;
    void InvalidatePositions(bool canonical_geometry = false);
    bool HasDeformationHistory() const;
    void BindPreviousPositions(mtl::Command &) const;
    void CapturePositions(mtl::PassChain &, const GpuBuffers &);
    bool PreviousCameraInsideMedium() const;
    uint32_t InstanceCount() const;
    uint64_t Revision() const;
    uint64_t GeometryRevision() const;

private:
    struct Storage;
    std::unique_ptr<Storage> Data;
};
