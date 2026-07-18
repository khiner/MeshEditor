#pragma once

#include "render/Bindless.h"

#include <entt/entity/fwd.hpp>

#include <array>
#include <functional>

struct DrawListBuilder;
struct SelectionDrawInfo;

using SelectionBuildFn = std::function<std::vector<SelectionDrawInfo>(DrawListBuilder &)>;

// RAII for the descriptor-slot leases used by the selection compute/render pipeline.
struct SelectionSlots {
    uint32_t HeadImage{}, SelectionCounter{}, ObjectPickKey{}, ElementPickCandidates{}, ObjectPickSeenBits{}, SelectionBitset{};
    uint32_t MotionBlurTileImage{}, MotionBlurTileIndirection{};
    uint32_t ObjectIdSampler{}, DepthSampler{}, SilhouetteSampler{}, SceneColorSampler{}, OverlayColorSampler{}, LineDataSampler{}, TransmissionSampler{}, MotionBlurAccumSampler{}, SceneDepthSampler{}, VelocitySampler{}, MotionBlurGatherSampler{}, DepthPyramidSampler{};

    using Entry = std::pair<SlotType, uint32_t SelectionSlots::*>;
    static constexpr std::array Entries{
        Entry{SlotType::Image, &SelectionSlots::HeadImage},
        Entry{SlotType::Buffer, &SelectionSlots::SelectionCounter},
        Entry{SlotType::Buffer, &SelectionSlots::ObjectPickKey},
        Entry{SlotType::Buffer, &SelectionSlots::ElementPickCandidates},
        Entry{SlotType::Buffer, &SelectionSlots::ObjectPickSeenBits},
        Entry{SlotType::Buffer, &SelectionSlots::SelectionBitset},
        Entry{SlotType::Image, &SelectionSlots::MotionBlurTileImage},
        Entry{SlotType::Buffer, &SelectionSlots::MotionBlurTileIndirection},
        Entry{SlotType::Sampler, &SelectionSlots::ObjectIdSampler},
        Entry{SlotType::Sampler, &SelectionSlots::DepthSampler},
        Entry{SlotType::Sampler, &SelectionSlots::SilhouetteSampler},
        Entry{SlotType::Sampler, &SelectionSlots::SceneColorSampler},
        Entry{SlotType::Sampler, &SelectionSlots::OverlayColorSampler},
        Entry{SlotType::Sampler, &SelectionSlots::LineDataSampler},
        Entry{SlotType::Sampler, &SelectionSlots::TransmissionSampler},
        Entry{SlotType::Sampler, &SelectionSlots::MotionBlurAccumSampler},
        Entry{SlotType::Sampler, &SelectionSlots::SceneDepthSampler},
        Entry{SlotType::Sampler, &SelectionSlots::VelocitySampler},
        Entry{SlotType::Sampler, &SelectionSlots::MotionBlurGatherSampler},
        Entry{SlotType::Sampler, &SelectionSlots::DepthPyramidSampler},
    };

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
