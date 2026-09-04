#include "metal/Buffer.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <format>
#include <stdexcept>

namespace mtl {
namespace {
uint64_t NextPowerOfTwo(uint64_t x) {
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}
} // namespace

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

Buffer::Buffer(BufferContext &ctx, std::span<const std::byte> data) : Buffer(ctx, data.size()) { Update(data); }

Buffer::Buffer(Buffer &&other) noexcept
    : Ctx(other.Ctx), Slot(other.Slot), UsedSize(other.UsedSize),
      DeviceBuffer(std::move(other.DeviceBuffer)), Type(other.Type) {
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
    auto *mapped = static_cast<std::byte *>(DeviceBuffer->contents());
    std::memmove(mapped + to, mapped + from, size);
}

std::span<std::byte> Buffer::GetMutableRange(uint64_t offset, uint64_t size) const {
    if (!DeviceBuffer) return {};
    return {static_cast<std::byte *>(DeviceBuffer->contents()) + offset, size};
}

void Buffer::Reserve(uint64_t required_size) {
    if (DeviceBuffer && required_size <= DeviceBuffer->length()) return;
    const auto new_size = NextPowerOfTwo(required_size);
    auto new_device = NewBuffer(Ctx.Ctx, new_size);
    if (UsedSize > 0 && DeviceBuffer) std::memcpy(new_device->contents(), DeviceBuffer->contents(), UsedSize);
    if (DeviceBuffer) {
        Ctx.Ctx.RemoveResident(DeviceBuffer.get());
        Ctx.Retired.emplace_back(std::move(DeviceBuffer));
    }
    DeviceBuffer = std::move(new_device);
    UpdateSlot();
}

void Buffer::Update(std::span<const std::byte> data, uint64_t offset) {
    if (data.empty()) return;
    const auto required_size = offset + data.size();
    Reserve(required_size);
    UsedSize = std::max(UsedSize, required_size);
    std::memcpy(static_cast<std::byte *>(DeviceBuffer->contents()) + offset, data.data(), data.size());
}
} // namespace mtl
