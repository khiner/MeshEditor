#pragma once

#include "metal/Bindless.h"

#include <entt/entity/fwd.hpp>

#include <array>
struct SelectionSlots {
    uint32_t ObjectPickKey{}, ElementPickKey{}, ElementPickId{}, ObjectPickSeenBits{}, ObjectBoxBitset{};
    uint32_t MotionBlurTileImage{}, MotionBlurTileIndirection{};
    uint32_t ObjectIdSampler{}, DepthSampler{}, SilhouetteSampler{}, SceneColorSampler{}, OverlayColorSampler{}, LineDataSampler{}, TransmissionSampler{}, MotionBlurAccumSampler{}, SceneDepthSampler{}, VelocitySampler{}, MotionBlurGatherSampler{}, DepthPyramidSampler{};

    using Entry = std::pair<SlotType, uint32_t SelectionSlots::*>;
    static constexpr std::array Entries{
        Entry{SlotType::Buffer, &SelectionSlots::ObjectPickKey},
        Entry{SlotType::Buffer, &SelectionSlots::ElementPickKey},
        Entry{SlotType::Buffer, &SelectionSlots::ElementPickId},
        Entry{SlotType::Buffer, &SelectionSlots::ObjectPickSeenBits},
        Entry{SlotType::Buffer, &SelectionSlots::ObjectBoxBitset},
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

    explicit SelectionSlots(mtl::BindlessSet &slots) : Slots(&slots) {
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
    mtl::BindlessSet *Slots{nullptr};
    void Release() {
        if (!Slots) return;
        for (const auto &[type, field] : Entries) Slots->Release({type, this->*field});
        Slots = nullptr;
    }
};
