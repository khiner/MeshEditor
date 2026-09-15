#include "metal/Buffer.h"

#include "metal/MetalCpp.h"
#include "project/BufferHistory.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <format>
#include <stdexcept>

namespace mtl {
BufferContext::BufferContext(const Context &ctx, BindlessSet &slots) : Ctx(ctx), Slots(slots) {}
BufferContext::~BufferContext() = default;
BufferContext::BufferContext(const BufferContext &) = default;
BufferContext::BufferContext(BufferContext &&) noexcept = default;
void BufferContext::ReclaimRetiredBuffers() { Retired.clear(); }

NS::SharedPtr<MTL::Buffer> NewBuffer(const Context &ctx, uint64_t size) {
    if (size == 0) return {};
    auto buffer = NS::TransferPtr(ctx.Device->newBuffer(size, MTL::ResourceStorageModeShared));
    if (!buffer) throw std::runtime_error("Failed to allocate a Metal buffer.");
    return buffer;
}

std::string BufferContext::DebugHeapUsage() const {
    static constexpr std::array<std::string_view, 6> Suffixes{"B", "KB", "MB", "GB", "TB", "PB"};
    const auto format_bytes = [](uint64_t bytes) {
        auto value = float(bytes);
        size_t pow = 0;
        for (; value >= 1024.f && pow + 1 < Suffixes.size(); ++pow) value /= 1024.f;
        return std::format("{:.2f} {}", value, Suffixes[pow]);
    };
    return std::format(
        "Device allocation:\n\tCurrent: {}\n\tRecommended maximum: {}\n",
        format_bytes(Ctx.Device->currentAllocatedSize()), format_bytes(Ctx.Device->recommendedMaxWorkingSetSize())
    );
}

Buffer::Buffer(BufferContext &ctx, uint64_t size, SlotType slot_type)
    : Ctx(ctx), Slot(ctx.Slots.Allocate(slot_type)), DeviceBuffer(NewBuffer(ctx.Ctx, size)), Type(slot_type) {
    if (size > 0) UpdateSlot();
}

Buffer::Buffer(BufferContext &ctx, std::span<const std::byte> data, SlotType slot_type)
    : Buffer(ctx, data.size(), slot_type) { Update(data); }

Buffer::Buffer(BufferContext &ctx, uint64_t size) : Ctx(ctx), DeviceBuffer(NewBuffer(ctx.Ctx, size)) {}

Buffer::Buffer(Buffer &&other) noexcept
    : Ctx(other.Ctx), Slot(other.Slot), UsedSize(other.UsedSize),
      DeviceBuffer(std::move(other.DeviceBuffer)), Tracked(std::move(other.Tracked)), Type(other.Type) {
    if (Tracked) Tracked->B = this;
    other.Slot = InvalidSlot;
}

Buffer &Buffer::operator=(Buffer &&other) noexcept {
    if (this != &other) {
        Retire();
        if (Slot != InvalidSlot) Ctx.Slots.Release({Type, Slot});
        Slot = other.Slot;
        UsedSize = other.UsedSize;
        DeviceBuffer = std::move(other.DeviceBuffer);
        Type = other.Type;
        Tracked = std::move(other.Tracked);
        if (Tracked) Tracked->B = this;
        other.Slot = InvalidSlot;
    }
    return *this;
}

Buffer::~Buffer() {
    Retire();
    if (Slot != InvalidSlot) Ctx.Slots.Release({Type, Slot});
}

void Buffer::Retire() {
    if (!DeviceBuffer) return;
    Ctx.Ctx.RemoveResident(DeviceBuffer.get());
    Ctx.Retired.emplace_back(std::move(DeviceBuffer));
}

void Buffer::UpdateSlot() {
    if (Slot == InvalidSlot) return;
    Ctx.Slots.SetBuffer({Type, Slot}, DeviceBuffer.get());
}

std::span<std::byte> Buffer::Contents() const {
    if (!DeviceBuffer) return {};
    return {static_cast<std::byte *>(DeviceBuffer->contents()), DeviceBuffer->length()};
}

void Buffer::Move(uint64_t from, uint64_t to, uint64_t size) const {
    if (!DeviceBuffer) return;
    const auto allocated = DeviceBuffer->length();
    if (size == 0 || from + size > allocated || to + size > allocated) return;
    CaptureWrite(to, size);
    auto *mapped = static_cast<std::byte *>(DeviceBuffer->contents());
    std::memmove(mapped + to, mapped + from, size);
}

std::span<std::byte> Buffer::GetMutableRange(uint64_t offset, uint64_t size) const {
    if (!DeviceBuffer) return {};
    CaptureWrite(offset, size);
    return {static_cast<std::byte *>(DeviceBuffer->contents()) + offset, size};
}

void Buffer::Reserve(uint64_t required_size) {
    if (required_size == 0) return;
    if (DeviceBuffer && required_size <= DeviceBuffer->length()) return;
    const auto new_size = std::bit_ceil(required_size);
    auto new_device = NewBuffer(Ctx.Ctx, new_size);
    if (Tracked) std::memset(new_device->contents(), 0, new_size);
    if (DeviceBuffer) {
        // Preserve pages written by cold restoration before UsedSize was updated.
        const auto preserved = Tracked ? DeviceBuffer->length() : UsedSize;
        if (preserved) std::memcpy(new_device->contents(), DeviceBuffer->contents(), preserved);
    }
    Retire();
    DeviceBuffer = std::move(new_device);
    UpdateSlot();
}

void Buffer::Update(std::span<const std::byte> data, uint64_t offset) {
    if (data.empty()) return;
    const auto required_size = offset + data.size();
    CaptureWrite(offset, data.size());
    SetUsedSize(std::max(UsedSize, required_size));
    std::memcpy(static_cast<std::byte *>(DeviceBuffer->contents()) + offset, data.data(), data.size());
}

void Buffer::CaptureWrite(uint64_t offset, uint64_t size) const {
    if (Tracked) Tracked->Write(offset, size);
}

void Buffer::SetUsedSize(uint64_t size) {
    Reserve(size);
    if (Tracked && size != UsedSize) {
        const auto first = std::min(size, UsedSize);
        const auto count = std::max(size, UsedSize) - first;
        CaptureWrite(first, count);
        if (size < UsedSize) std::memset(static_cast<std::byte *>(DeviceBuffer->contents()) + size, 0, count);
    }
    UsedSize = size;
}

void Buffer::Track(store::History &history, std::string name, uint32_t page_bytes) {
    Tracked = std::make_unique<project::BufferHistory>(*this, history, std::move(name), page_bytes);
}
} // namespace mtl
