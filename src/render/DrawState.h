#pragma once

#include "gpu/DrawData.h"

#include <entt/entity/entity.hpp>

#include <optional>
#include <unordered_map>
#include <vector>

// One draw within a batch: where its first instance's draw data sits relative to the batch, how many
// instances follow it, and how many indices each instance walks.
struct DrawRecord {
    uint32_t FirstDraw{0}, InstanceCount{0}, IndexCount{0};
};

struct DrawBatchInfo {
    uint32_t DrawDataSlotOffset{0}, DrawCount{0}, FirstRecord{0};
};

struct DrawListBuilder {
    std::vector<DrawData> Draws;
    std::vector<DrawRecord> Records;

    DrawBatchInfo BeginBatch() const { return {uint32_t(Draws.size()), 0, uint32_t(Records.size())}; }

    void Append(DrawBatchInfo &batch, const DrawData &draw, uint32_t index_count, uint32_t instance_count) {
        if (index_count == 0 || instance_count == 0) return;
        const uint32_t draw_data_start = Draws.size();
        for (uint32_t i = 0; i < instance_count; ++i) {
            DrawData per_instance = draw;
            per_instance.FirstInstance = draw.FirstInstance + i;
            Draws.emplace_back(per_instance);
        }
        Records.push_back({draw_data_start - batch.DrawDataSlotOffset, instance_count, index_count});
        ++batch.DrawCount;
    }
};

// Posed-buffer layout for one mesh entity's instance run: base offsets plus per-instance strides.
struct PosedRanges {
    // One instance's derived normals, absent for a mesh with no faces to derive them from.
    struct NormalRanges {
        uint32_t VertexOffset{InvalidOffset}, SeamOffset{InvalidOffset}, FaceOffset{InvalidOffset};
        uint32_t SeamCount{0}, FaceCount{0};
    };

    uint32_t FirstInstance{0};
    bool PerInstance{false};
    uint32_t PositionBase{InvalidOffset};
    uint32_t VertexCount{0};
    uint32_t MeshletBoundsBase{InvalidOffset};
    uint32_t Level0Count{0}; // Original clusters across the mesh's primitives, which is what posed bounds cover.
    std::optional<NormalRanges> Normals{}; // Instance 0's offsets.

    uint32_t PositionOffset(uint32_t i) const { return PositionBase + i * VertexCount; }
    uint32_t MeshletBoundsOffset(uint32_t i) const { return MeshletBoundsBase + i * Level0Count; }
    std::optional<NormalRanges> NormalsAt(uint32_t i) const {
        if (!Normals) return std::nullopt;
        return NormalRanges{
            Normals->VertexOffset + i * VertexCount,
            Normals->SeamOffset + i * Normals->SeamCount,
            Normals->FaceOffset + i * Normals->FaceCount,
            Normals->SeamCount,
            Normals->FaceCount,
        };
    }
};

// The excitable vertex handles one sound-point dispatch draws.
// The strike and active handles change between draw list rebuilds, so recording reads them fresh.
struct SoundPointInfo {
    entt::entity InstanceEntity{entt::null};
    uint32_t VertexOffset{}, VertexCount{};
};

// Persistent draw data, rebuilt when scene structure or batch routing changes.
struct DrawState {
    DrawListBuilder List;
    std::unordered_map<entt::entity, PosedRanges> PosedByEntity;
    DrawBatchInfo EdgeQuad, WireLine, WireMeshlet, Point, SoundPoint;
    DrawBatchInfo BoneFill, BoneWire, BoneSphereFill, BoneSphereWire;
    DrawBatchInfo OverlayFaceNormals, OverlayVertexNormals;
    std::vector<SoundPointInfo> SoundPoints; // Parallel to the SoundPoint batch's draw records.
    DrawBatchInfo SelectionBoneSpheres; // Bone joint picks, emitted per bone instance.
    DrawBatchInfo SelectionLines, SelectionPoints; // Line and point mesh picks.

    // Cached selection pass draw list, reused when only the camera changed.
    DrawListBuilder SelectionList;
    bool SelectionStale{true}; // Selection fragment data no longer matches the scene. Cleared after RenderSelectionPass.

    bool InstanceRecordsStale{true}; // A record field other than the object id or flags needs a rewrite.
    bool InstanceFlagsStale{true}; // Object ids and silhouette flags need a rewrite.
    uint64_t InstanceRecordInputs{0}; // Signature of the mesh-keyed inputs the records read.
};

// Mark every instance record for a rewrite, for a change the record-input signature does not see.
inline void MarkInstanceRecordsStale(DrawState &draw) {
    draw.InstanceRecordsStale = true;
    draw.InstanceFlagsStale = true;
}
