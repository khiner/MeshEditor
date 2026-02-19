#pragma once

#include "gpu/DrawData.h"

struct DrawBatchInfo {
    uint32_t DrawDataSlotOffset{0};
    uint32_t DrawCount{0};
    vk::DeviceSize IndirectOffset{0};
};

struct DrawListBuilder {
    std::vector<DrawData> Draws;
    std::vector<vk::DrawIndexedIndirectCommand> IndirectCommands;
    uint32_t MaxIndexCount{0};

    DrawBatchInfo BeginBatch() {
        return {uint32_t(Draws.size()), 0, IndirectCommands.size() * sizeof(vk::DrawIndexedIndirectCommand)};
    }

    void Append(DrawBatchInfo &batch, const DrawData &draw, uint32_t index_count, uint32_t instance_count) {
        if (index_count == 0 || instance_count == 0) return;
        const auto draw_data_start = uint32_t(Draws.size());
        Draws.reserve(Draws.size() + instance_count);
        for (uint32_t i = 0; i < instance_count; ++i) {
            DrawData per_instance = draw;
            per_instance.FirstInstance = draw.FirstInstance + i;
            Draws.emplace_back(per_instance);
        }
        const uint32_t first_instance = draw_data_start - batch.DrawDataSlotOffset;
        IndirectCommands.emplace_back(vk::DrawIndexedIndirectCommand{index_count, instance_count, 0, 0, first_instance});
        MaxIndexCount = std::max(MaxIndexCount, index_count);
        ++batch.DrawCount;
    }
};

struct SelectionDrawInfo {
    ShaderPipelineType Pipeline{ShaderPipelineType::Fill};
    DrawBatchInfo Batch{};
};

#include "gpu/DrawPassPushConstants.h"

namespace {
// If `model_index` is set, only the model at that index is rendered. Otherwise, all models are rendered.
void AppendDraw(
    DrawListBuilder &builder, DrawBatchInfo &batch, uint32_t index_count, const ModelsBuffer &models,
    DrawData draw, std::optional<uint> model_index = {}
) {
    draw.FirstInstance = model_index.value_or(0);
    const auto instance_count = model_index.has_value() ? 1 : models.Buffer.UsedSize / sizeof(WorldTransform);
    builder.Append(batch, draw, index_count, uint32_t(instance_count));
}

void AppendDraw(
    DrawListBuilder &builder, DrawBatchInfo &batch, const SlottedRange &indices, const ModelsBuffer &models,
    DrawData draw, std::optional<uint> model_index = {}
) {
    AppendDraw(builder, batch, indices.Count, models, draw, model_index);
}

DrawData MakeDrawData(
    uint32_t vertex_slot,
    Range vertices,
    const SlottedRange &indices,
    uint32_t model_slot,
    uint32_t instance_state_slot = InvalidSlot,
    uint32_t bone_deform = InvalidOffset,
    uint32_t armature_deform = InvalidOffset,
    uint32_t morph_deform = InvalidOffset,
    uint32_t morph_target_count = 0
) {
    return {
        .VertexSlot = vertex_slot,
        .IndexSlotOffset = indices,
        .ModelSlot = model_slot,
        .ObjectIdSlot = InvalidSlot,
        .VertexCountOrHeadImageSlot = vertices.Count,
        .InstanceStateSlot = instance_state_slot,
        .VertexOffset = vertices.Offset,
        .BoneDeformOffset = bone_deform,
        .ArmatureDeformOffset = armature_deform,
        .MorphDeformOffset = morph_deform,
        .MorphTargetCount = morph_target_count,
    };
}
DrawData MakeDrawData(const SlottedRange &vertices, const SlottedRange &indices, const ModelsBuffer &mb, uint32_t bone_deform = InvalidOffset, uint32_t armature_deform = InvalidOffset, uint32_t morph_deform = InvalidOffset, uint32_t morph_target_count = 0) {
    return MakeDrawData(vertices.Slot, vertices, indices, mb.Buffer.Slot, mb.InstanceStates.Slot, bone_deform, armature_deform, morph_deform, morph_target_count);
}
DrawData MakeDrawData(const RenderBuffers &rb, uint32_t vertex_slot, const ModelsBuffer &mb) {
    return MakeDrawData(vertex_slot, rb.Vertices, rb.Indices, mb.Buffer.Slot);
}
} // namespace
