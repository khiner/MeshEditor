#pragma once

#include "metal/MetalContext.h"

#include <span>

namespace mtl {
// One color attachment at record time, with the load and store actions Metal picks per pass.
struct ColorAttachment {
    MTL::Texture *Texture{nullptr};
    MTL::LoadAction Load{MTL::LoadActionLoad};
    MTL::StoreAction Store{MTL::StoreActionStore};
    MTL::ClearColor Clear{0, 0, 0, 0};
    uint32_t Level{0};
};

struct DepthAttachment {
    MTL::Texture *Texture{nullptr};
    MTL::LoadAction Load{MTL::LoadActionLoad};
    MTL::StoreAction Store{MTL::StoreActionStore};
    double Clear{1.0};
};

MTL::RenderPassDescriptor *MakePassDescriptor(std::span<const ColorAttachment>, DepthAttachment = {});

inline ColorAttachment ClearColor(MTL::Texture *texture, MTL::ClearColor clear = {0, 0, 0, 0}) {
    return {texture, MTL::LoadActionClear, MTL::StoreActionStore, clear};
}
inline ColorAttachment LoadColor(MTL::Texture *texture) {
    return {texture, MTL::LoadActionLoad, MTL::StoreActionStore};
}
inline ColorAttachment DiscardColor(MTL::Texture *texture) {
    return {texture, MTL::LoadActionDontCare, MTL::StoreActionStore};
}
inline DepthAttachment ClearDepth(MTL::Texture *texture, double clear = 1.0) {
    return {texture, MTL::LoadActionClear, MTL::StoreActionStore, clear};
}
inline DepthAttachment LoadDepth(MTL::Texture *texture, MTL::StoreAction store = MTL::StoreActionStore) {
    return {texture, MTL::LoadActionLoad, store};
}
} // namespace mtl
