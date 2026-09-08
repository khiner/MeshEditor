#pragma once

#include <string>
#include <type_traits>

#include <Foundation/NSSharedPtr.hpp>

namespace NS {
class String;
}
namespace MTL {
class Allocation;
class Device;
class CommandQueue;
class ResidencySet;
} // namespace MTL

namespace mtl {
NS::String *Str(std::string_view);

// Prefer queue-wide residency sets; BindlessSet declares its resources per encoder when unavailable.
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
