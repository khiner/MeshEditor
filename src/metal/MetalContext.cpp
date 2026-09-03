#include "metal/MetalContext.h"

#include <stdexcept>
#include <utility>

namespace mtl {
NS::String *Str(std::string_view s) {
    return NS::String::string(std::string{s}.c_str(), NS::UTF8StringEncoding);
}

Context::Context() {
    Device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!Device) throw std::runtime_error("No Metal device.");
    if (!Device->hasUnifiedMemory()) throw std::runtime_error("MeshEditor targets unified-memory Apple Silicon.");
    if (Device->argumentBuffersSupport() < MTL::ArgumentBuffersTier2) {
        throw std::runtime_error("The bindless argument buffer needs argument buffer tier 2.");
    }

    Queue = NS::TransferPtr(Device->newCommandQueue());
    if (!Queue) throw std::runtime_error("Failed to create a Metal command queue.");

    const auto descriptor = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
    Residency = NS::TransferPtr(Device->newResidencySet(descriptor.get(), nullptr));
    if (Residency) {
        Residency->commit();
        Queue->addResidencySet(Residency.get());
    }
}

Context::Context(Context &&) noexcept = default;

Context::~Context() {
    if (Queue && Residency) Queue->removeResidencySet(Residency.get());
}

void Context::AddResident(MTL::Allocation *resource) const {
    if (!Residency || !resource) return;
    Residency->addAllocation(resource);
    ResidencyDirty = true;
}

void Context::RemoveResident(MTL::Allocation *resource) const {
    if (!Residency || !resource) return;
    Residency->removeAllocation(resource);
    ResidencyDirty = true;
}

void Context::CommitResidency() const {
    if (!Residency || !std::exchange(ResidencyDirty, false)) return;
    Residency->commit();
}
}
