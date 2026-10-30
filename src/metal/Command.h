#pragma once

#include "metal/MetalContext.h"

#include <Metal/MTL4ComputeCommandEncoder.hpp>
#include <Metal/MTL4RenderCommandEncoder.hpp>

#include <array>

namespace MTL {
class Texture;
class SamplerState;
class DepthStencilState;
} // namespace MTL
namespace MTL4 {
class RenderPassDescriptor;
} // namespace MTL4

namespace mtl {
// An encoder of the command whose argument table its draws and dispatches snapshot.
template<typename Native> struct Encoder {
    Command *Cmd{};
    Native *Pass{};

    Native *operator->() const { return Pass; }
    explicit operator bool() const { return Pass != nullptr; }
};
using ComputeEncoder = Encoder<MTL4::ComputeCommandEncoder>;
using RenderEncoder = Encoder<MTL4::RenderCommandEncoder>;

// Orders the encoder's later dispatches and copies after its earlier ones.
inline void ComputeBarrier(ComputeEncoder encoder) {
    encoder->barrierAfterEncoderStages(MTL::StageDispatch | MTL::StageBlit, MTL::StageDispatch | MTL::StageBlit, MTL4::VisibilityOptionDevice);
}
// Records the cull mode when it differs from the one the encoder holds.
void SetCullMode(RenderEncoder, MTL::CullMode);
// Records the threadgroup memory length at `index` when it differs from the one the encoder holds.
void SetThreadgroupMemoryLength(ComputeEncoder, uint32_t length, uint32_t index);

// One command buffer recording, with the argument tables its encoders bind and storage for the bytes they upload.
// Bindings set through the command apply to the draws and dispatches encoded after them, on every encoder of the command.
// Compute, vertex, object and mesh stages share one table, and the fragment stage has its own, so a pass can give its fragment function separate constants.
// Commit it through the context, which returns the serial its completion signals.
// An uncommitted recording ends without executing when destroyed.
struct Command {
    ~Command();
    Command(const Command &) = delete;
    Command &operator=(const Command &) = delete;
    Command(Command &&) noexcept;
    Command &operator=(Command &&) = delete;

    MTL4::CommandBuffer *operator*() const;
    MTL4::ArgumentTable *Arguments() const;
    MTL4::ArgumentTable *FragmentArguments() const;
    // Begins an encoder with the argument table bound.
    ComputeEncoder BeginCompute();
    RenderEncoder BeginRender(const MTL4::RenderPassDescriptor *);

    // Copies bytes into storage the command keeps until it completes, returning their GPU address.
    uint64_t Upload(std::span<const std::byte>);
    void SetBytes(std::span<const std::byte> bytes, uint32_t index);
    void SetBytes(const auto &value, uint32_t index) { SetBytes(std::span{reinterpret_cast<const std::byte *>(&value), sizeof(value)}, index); }
    // Binds for the compute, vertex, object and mesh stages alone.
    void SetGeometryBytes(std::span<const std::byte> bytes, uint32_t index);
    void SetGeometryBytes(const auto &value, uint32_t index) { SetGeometryBytes(std::span{reinterpret_cast<const std::byte *>(&value), sizeof(value)}, index); }
    // Binds for the fragment stage alone.
    void SetFragmentBytes(std::span<const std::byte> bytes, uint32_t index);
    void SetFragmentBytes(const auto &value, uint32_t index) { SetFragmentBytes(std::span{reinterpret_cast<const std::byte *>(&value), sizeof(value)}, index); }
    void SetAddress(uint64_t address, uint32_t index);
    void SetBuffer(const MTL::Buffer *, uint64_t offset, uint32_t index);
    void SetTexture(const MTL::Texture *, uint32_t index);
    void SetSampler(const MTL::SamplerState *, uint32_t index);

    // The commit waits for the event on the queue before the command, and signals it after.
    void WaitEvent(MTL::Event *, uint64_t value);
    void SignalEvent(MTL::Event *, uint64_t value);
    // The commit waits for the display to release the drawable before the command, and presents it after.
    void Present(MTL::Drawable *);

    // The state the active encoder holds, which pipeline binds and the Set functions above record only when it changes.
    struct EncoderState {
        const NS::Object *Pipeline{};
        const MTL::DepthStencilState *DepthStencil{};
        MTL::CullMode Cull{MTL::CullModeNone};
        std::array<uint32_t, 32> ThreadgroupLengths{};
    };
    EncoderState State;

    const Context &Ctx;

private:
    friend struct Context;
    Command(const Context &, uint32_t slot);
    Context::CommandSlot &Slot() const;
    uint32_t Index;
    bool Live{true};
};
} // namespace mtl
