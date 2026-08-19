#include "metal/MetalContext.h"

#include <format>
#include <stdexcept>
#include <utility>

namespace mtl {
NS::String *Str(std::string_view s) {
    return NS::String::string(std::string{s}.c_str(), NS::UTF8StringEncoding);
}

Context::Context() {
    Device = Owned<MTL::Device>{MTL::CreateSystemDefaultDevice()};
    if (!Device) throw std::runtime_error("No Metal device.");
    if (!Device->hasUnifiedMemory()) throw std::runtime_error("MeshEditor targets unified-memory Apple Silicon.");
    if (Device->argumentBuffersSupport() < MTL::ArgumentBuffersTier2) {
        throw std::runtime_error("The bindless argument buffer needs argument buffer tier 2.");
    }

    Queue = Owned<MTL::CommandQueue>{Device->newCommandQueue()};
    if (!Queue) throw std::runtime_error("Failed to create a Metal command queue.");

    const Owned<MTL::ResidencySetDescriptor> descriptor{MTL::ResidencySetDescriptor::alloc()->init()};
    NS::Error *error = nullptr;
    Residency = Owned<MTL::ResidencySet>{Device->newResidencySet(*descriptor, &error)};
    if (!Residency) {
        throw std::runtime_error(std::format("Failed to create a residency set: {}", error ? error->localizedDescription()->utf8String() : "unknown"));
    }
    Residency->commit();
    Queue->addResidencySet(*Residency);
}

Context::Context(Context &&other) noexcept
    : Device(std::move(other.Device)), Queue(std::move(other.Queue)), Residency(std::move(other.Residency)),
      ResidencyDirty(other.ResidencyDirty) {}

Context::~Context() {
    if (Queue && Residency) Queue->removeResidencySet(*Residency);
}

void Context::AddResident(MTL::Allocation *resource) const {
    if (!resource) return;
    Residency->addAllocation(resource);
    ResidencyDirty = true;
}

void Context::RemoveResident(MTL::Allocation *resource) const {
    if (!resource) return;
    Residency->removeAllocation(resource);
    ResidencyDirty = true;
}

void Context::CommitResidency() const {
    if (!std::exchange(ResidencyDirty, false)) return;
    Residency->commit();
}
} // namespace mtl
