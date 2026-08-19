#pragma once

#include "metal/MetalCpp.h"

#include <string>

namespace mtl {
NS::String *Str(std::string_view);

// Metal requires anything reached through an argument buffer to be resident, so allocations register
// here once rather than per encoder.
struct Context {
    Context();
    ~Context();
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;
    Context(Context &&) noexcept;
    Context &operator=(Context &&) = delete;

    void AddResident(MTL::Allocation *) const;
    void RemoveResident(MTL::Allocation *) const;
    void CommitResidency() const;

    NS::SharedPtr<MTL::Device> Device;
    NS::SharedPtr<MTL::CommandQueue> Queue;
    NS::SharedPtr<MTL::ResidencySet> Residency;

private:
    mutable bool ResidencyDirty{false};
};
} // namespace mtl
