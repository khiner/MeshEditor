#pragma once

#include "gpu/PBRMaterial.h"
#include "numeric/mat4.h"
#include "vulkan/Range.h"

#include <string>
#include <vector>

struct MaterialStore {
    std::vector<std::string> Names;
};

struct SceneBuffers;

// Material buffer accessors (SceneBuffers must be complete at call sites).
uint32_t GetMaterialCount(const SceneBuffers &);
const PBRMaterial &GetMaterial(const SceneBuffers &, uint32_t index);
PBRMaterial &GetMaterial(SceneBuffers &, uint32_t index);
void SetMaterial(SceneBuffers &, uint32_t index, const PBRMaterial &);
uint32_t AppendMaterial(SceneBuffers &, const PBRMaterial &);
void SetMaterialCount(SceneBuffers &, uint32_t count);

// Buffer context and arena accessors (SceneBuffers* only needs to be declared, not complete).
namespace mvk {
struct BufferContext;
}
mvk::BufferContext &GetBufferCtx(SceneBuffers *);
Range AllocateArmatureDeform(SceneBuffers *, uint32_t count);
std::span<mat4> GetArmatureDeformMutable(SceneBuffers *, Range);
Range AllocateMorphWeights(SceneBuffers *, uint32_t count);
std::span<float> GetMorphWeightsMutable(SceneBuffers *, Range);
