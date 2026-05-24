#pragma once

#include "Bindless.h"
#include "BoneSelection.h"
#include "numeric/vec2.h"

// A contiguous span of a mesh's elements (vertices/edges/faces) in the SelectionBitset.
struct ElementRange {
    entt::entity MeshEntity;
    uint32_t Offset, Count;
};

// Snapshot of selection state at the start of a shift+box-drag.
// Presence on viewport means an additive box-drag is active.
struct AdditiveBoxSelectBaseline {
    std::vector<entt::entity> SelectedEntities;
    std::vector<std::pair<entt::entity, BoneSelection>> BoneSelections;
    std::vector<uint32_t> ElementBitset;
};

// Flags on viewport, consumed and cleared by ProcessComponentEvents.
struct SelectionBitsDirty {}; // Bitset written by Interact; dispatches the compute update.
struct ElementStatesDirty {}; // Element state buffers updated by GPU compute; triggers a submit.
struct SelectionStale {}; // Selection fragment data no longer matches current scene. Cleared after RenderSelectionPass.

struct PendingEditElementClick {
    uvec2 MousePx;
    bool Toggle;
};

// Non-owning span over the GPU-mapped SelectionBitset words.
struct SelectionBitsetRef {
    std::span<uint32_t> Value;
};

// RAII for the descriptor-slot leases used by the selection compute/render pipeline.
struct SelectionSlots {
    uint32_t HeadImage{}, SelectionCounter{}, ObjectPickKey{}, ElementPickCandidates{}, ObjectPickSeenBits{}, SelectionBitset{};
    uint32_t ObjectIdSampler{}, DepthSampler{}, SilhouetteSampler{}, ColorSampler{}, LineDataSampler{}, TransmissionSampler{};

    using Entry = std::pair<SlotType, uint32_t SelectionSlots::*>;
    static constexpr std::array<Entry, 12> Entries{{
        {SlotType::Image, &SelectionSlots::HeadImage},
        {SlotType::Buffer, &SelectionSlots::SelectionCounter},
        {SlotType::Buffer, &SelectionSlots::ObjectPickKey},
        {SlotType::Buffer, &SelectionSlots::ElementPickCandidates},
        {SlotType::Buffer, &SelectionSlots::ObjectPickSeenBits},
        {SlotType::Buffer, &SelectionSlots::SelectionBitset},
        {SlotType::Sampler, &SelectionSlots::ObjectIdSampler},
        {SlotType::Sampler, &SelectionSlots::DepthSampler},
        {SlotType::Sampler, &SelectionSlots::SilhouetteSampler},
        {SlotType::Sampler, &SelectionSlots::ColorSampler},
        {SlotType::Sampler, &SelectionSlots::LineDataSampler},
        {SlotType::Sampler, &SelectionSlots::TransmissionSampler},
    }};

    explicit SelectionSlots(DescriptorSlots &slots) : Slots(&slots) {
        for (const auto &[type, field] : Entries) this->*field = slots.Allocate(type);
    }
    SelectionSlots(const SelectionSlots &) = delete;
    SelectionSlots &operator=(const SelectionSlots &) = delete;
    SelectionSlots(SelectionSlots &&o) noexcept : Slots(o.Slots) {
        for (const auto &[_, field] : Entries) this->*field = o.*field;
        o.Slots = nullptr;
    }
    SelectionSlots &operator=(SelectionSlots &&o) noexcept {
        if (this != &o) {
            Release();
            Slots = o.Slots;
            for (const auto &[_, field] : Entries) this->*field = o.*field;
            o.Slots = nullptr;
        }
        return *this;
    }
    ~SelectionSlots() { Release(); }

private:
    DescriptorSlots *Slots{nullptr};
    void Release() {
        if (!Slots) return;
        for (const auto &[type, field] : Entries) Slots->Release({type, this->*field});
        Slots = nullptr;
    }
};

// Selection ignores occlusion when true.
struct SelectionXRay {
    bool Value{false};
};

enum class SelectionGesture : uint8_t {
    Click,
    Box,
};

struct BoxSelectState {
    SelectionGesture Gesture{SelectionGesture::Box};
};

struct SelectedInstanceCount {
    uint32_t Value{0};
};
