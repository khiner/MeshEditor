#pragma once

#include "render/Bindless.h"
#include "selection/SelectionQueries.h"

#include <entt/entity/fwd.hpp>
#include <vulkan/vulkan.hpp>

#include <array>
#include <functional>

struct DrawListBuilder;
struct SelectionDrawInfo;

using SelectionBuildFn = std::function<std::vector<SelectionDrawInfo>(DrawListBuilder &)>;

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

// Render the on-demand selection-fragment pass. `build_fn` populates the draw list given the silhouette-prefilled builder.
void RenderSelectionPassWith(entt::registry &, entt::entity viewport, bool render_depth, const SelectionBuildFn &, vk::Semaphore signal_semaphore = {}, bool render_silhouette = true);

// Replays the cached selection draw list (built by RecordRenderCommandBuffer). Clears SelectionStale on success.
void RenderSelectionPass(entt::registry &, entt::entity viewport, vk::Semaphore signal_semaphore = {});
