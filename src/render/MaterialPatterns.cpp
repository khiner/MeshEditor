#include "render/MaterialPatterns.h"
#include "SortUnique.h"
#include "gpu/MeshAttributeBit.h"
#include "mesh/MeshStore.h"
#include "render/GpuBuffers.h"
#include "render/ObjectGeometry.h"

void Reset(MaterialPatterns &patterns, state::Registry &) {
    patterns.Records.reset();
    patterns.Count = 0u;
    patterns.Values.clear();
    patterns.Words = 0u;
}

std::span<const MaterialPattern> PatternRecords(const MaterialPatterns &patterns) {
    return patterns.Records ? patterns.Records->GetSpan<MaterialPattern>({0u, patterns.Count}) : std::span<const MaterialPattern>{};
}

PbrFeatureMask PatternFeatures(uint32_t inputs) {
    const auto any = [&](std::initializer_list<PbrInput> of) { return std::ranges::any_of(of, [&](PbrInput input) { return (inputs >> uint32_t(input)) & 1u; }); };
    PbrFeatureMask features = FeatureBit(PbrFeature::Patterns);
    if (any({PbrInput::Clearcoat, PbrInput::ClearcoatRoughness, PbrInput::ClearcoatNormal})) features |= PbrFeature::Clearcoat;
    if (any({PbrInput::SheenColor, PbrInput::SheenRoughness})) features |= PbrFeature::Sheen;
    if (any({PbrInput::Iridescence, PbrInput::IridescenceThickness})) features |= PbrFeature::Iridescence;
    if (any({PbrInput::AnisotropyStrength, PbrInput::AnisotropyRotation})) features |= PbrFeature::Anisotropy;
    return features;
}

uint32_t PatternLodAttributes(const MaterialPattern &pattern, bool authored_tangents) {
    constexpr uint32_t Texcoords = MeshAttributeBit_TexCoord0 | MeshAttributeBit_TexCoord1 | MeshAttributeBit_TexCoord2 | MeshAttributeBit_TexCoord3;
    const auto needs = pattern.Program.Needs;
    return (needs & OpNeedsTexcoords ? Texcoords : 0u) | (authored_tangents && (needs & OpNeedsTangent) ? uint32_t(MeshAttributeBit_Tangent) : 0u);
}

DrawnPatterns DrawnPatternFeatures(const MaterialPatterns &patterns, const MeshStore &meshes, const GpuBuffers &buffers, const SceneViewUBO &view) {
    if (!patterns.Records) return {};
    const auto instance_states = buffers.InstanceStatesFor(view);
    if (instance_states.empty()) return {};
    const auto instances = buffers.Instances.RecordBuffer.GetSpan<InstanceRecord>();
    std::vector<uint32_t> drawn;
    for (const auto slot : buffers.GpuInstanceSlots.GetSpan<uint32_t>()) {
        if (const auto mesh = instances[slot].Mesh; mesh != InvalidOffset && !(instance_states[slot] & InstanceStateHidden)) drawn.push_back(mesh);
    }
    SortUnique(drawn);
    const auto materials = buffers.MaterialsFor(view);
    const auto records = PatternRecords(patterns);
    DrawnPatterns drawn_patterns;
    uint32_t inputs = 0u;
    bool patterned = false;
    for (const auto mesh : drawn) {
        const auto *record = meshes.TryGet(mesh);
        if (!record || !record->TriangleCount) continue;
        const auto assignments = meshes.Arenas().PrimitiveMaterials.Get(record->PrimitiveMaterials);
        meshes.ForEachPrimitive(*record, [&](ElementHandle, const PrimitiveRecord &primitive) {
            if (primitive.Topology != 0u || !primitive.TriangleCount) return;
            const uint32_t index = primitive.PrimitiveIndex < assignments.size() ? assignments[primitive.PrimitiveIndex] : 0u;
            if (index >= records.size() || records[index].Program.Index == InvalidOffset) return;
            inputs |= records[index].Inputs;
            patterned = true;
            if (index < materials.size()) drawn_patterns.Transparent |= materials[index].AlphaMode == MaterialAlphaMode::Blend || materials[index].Transmission.Factor > 0.f;
        });
    }
    if (patterned) drawn_patterns.Features = PatternFeatures(inputs);
    return drawn_patterns;
}
