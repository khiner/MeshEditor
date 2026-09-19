#include "animation/Fields.h"

#include "CameraTypes.h"
#include "animation/MorphWeights.h"
#include "armature/ArmatureComponents.h"
#include "object/ObjectOps.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/LightComponents.h"
#include "render/MaterialComponents.h"
#include "render/MaterialTextureSlots.h"
#include "state/Scene.h"

#include <cmath>
#include <cstring>

namespace animation {
namespace {
// The components a channel can write, each a paged record addressed as bytes at an offset.
template<typename... Cs> struct TypeList {};
using ChannelComponents = TypeList<PosedLocal, BoneDelta, Perspective, Orthographic, PunctualLight, ImageLight, Visibility>;

// Calls `f.template operator()<C>()` for the channel component keyed by `key`. Returns whether one matched.
template<typename F> bool ForChannelComponent(state::TypeKey key, F &&f) {
    return [&]<typename... Cs>(TypeList<Cs...>) {
        return ((state::Key<Cs>() == key && (f.template operator()<Cs>(), true)) || ...);
    }(ChannelComponents{});
}

static_assert(offsetof(PosedLocal, Value) == 0 && offsetof(BoneDelta, Value) == 0, "Pose fields share Transform's offsets");

bool NearlyEqual(float a, float b) { return std::abs(a - b) <= 1e-5f * std::max({1.f, std::abs(a), std::abs(b)}); }
} // namespace

uint16_t TextureSlotOffset(uint8_t texture) {
    PBRMaterial m;
    return uint16_t(reinterpret_cast<const std::byte *>(&MaterialTextureSlots[texture].Get(m)) - reinterpret_cast<const std::byte *>(&m));
}

ChannelTarget WeightsTarget(const state::Scene &r, state::Entity e) {
    const auto *weights = r.try_get<const MorphWeightRange>(e);
    return {state::Key<MorphWeightRange>(), 0, 0, uint16_t(weights ? weights->Weights.Count : 0), ValueKind::Float};
}

std::array<ChannelTarget, 3> TransformTargets(const state::Scene &r, state::Entity e) {
    if (r.all_of<BoneDelta>(e)) return {Target<&BoneDelta::Value, &Transform::P>(), Target<&BoneDelta::Value, &Transform::R>(), Target<&BoneDelta::Value, &Transform::S>()};
    return {Target<&Transform::P>(), Target<&Transform::R>(), Target<&Transform::S>()};
}

bool IsChannelStore(state::TypeKey key) {
    return key == state::Key<MaterialStore>() || key == state::Key<MorphWeightRange>() || ForChannelComponent(key, []<typename> {});
}

bool ReadField(const state::Scene &r, state::Entity e, const ChannelTarget &target, std::span<float> out) {
    if (out.size() != target.Count) return false;
    const auto bytes = target.Count * sizeof(float);
    if (target.Component == state::Key<MaterialStore>()) {
        const auto materials = r.Context.get<const GpuBuffers>().Materials.GetSpan<PBRMaterial>();
        if (target.Index >= materials.size()) return false;
        std::memcpy(out.data(), reinterpret_cast<const std::byte *>(&materials[target.Index]) + target.Offset, bytes);
        return true;
    }
    if (target.Component == state::Key<MorphWeightRange>()) {
        const auto *weights = r.try_get<const MorphWeightRange>(e);
        const auto first = target.Offset / sizeof(float);
        if (!weights || first + target.Count > weights->Weights.Count) return false;
        std::ranges::copy(r.Context.get<const GpuBuffers>().MorphWeightBuffer.Get(weights->Weights).subspan(first, target.Count), out.begin());
        return true;
    }
    bool found = false;
    ForChannelComponent(target.Component, [&]<typename C> {
        const void *record = r.try_get<const C>(e);
        // A node without a pose reads its Transform.
        if constexpr (std::same_as<C, PosedLocal>) {
            if (!record) record = r.try_get<const Transform>(e);
        }
        // A node without a Visibility flag reads its Hidden state.
        if constexpr (std::same_as<C, Visibility>) {
            if (!record) {
                out[0] = r.all_of<Hidden>(e) ? 0.f : 1.f;
                found = true;
                return;
            }
        }
        if (!record) return;
        const auto *field = static_cast<const std::byte *>(record) + target.Offset;
        if (target.Kind == ValueKind::Bool) {
            uint32_t flag;
            std::memcpy(&flag, field, sizeof flag);
            out[0] = flag ? 1.f : 0.f;
        } else {
            std::memcpy(out.data(), field, bytes);
        }
        found = true;
    });
    return found;
}

void WriteField(state::Scene &r, state::Entity e, const ChannelTarget &target, std::span<const float> in) {
    if (in.size() != target.Count) return;
    const auto bytes = target.Count * sizeof(float);
    if (target.Component == state::Key<MaterialStore>()) {
        auto &materials = r.Context.get<GpuBuffers>().Materials;
        if (target.Index >= materials.Count<PBRMaterial>()) return;
        materials.Update(std::as_bytes(in), uint64_t(target.Index) * sizeof(PBRMaterial) + target.Offset);
        reactive(r, state::Change::Materials).emplace(e);
        return;
    }
    if (target.Component == state::Key<MorphWeightRange>()) {
        const auto *weights = r.try_get<const MorphWeightRange>(e);
        const auto first = target.Offset / sizeof(float);
        if (!weights || first + target.Count > weights->Weights.Count) return;
        std::ranges::copy(in, r.Context.get<GpuBuffers>().MorphWeightBuffer.GetMutable(weights->Weights).begin() + first);
        reactive(r, state::Change::MorphWeights).emplace(e);
        return;
    }
    ForChannelComponent(target.Component, [&]<typename C> {
        if constexpr (std::same_as<C, Visibility>) {
            if (!r.all_of<C>(e)) r.emplace<C>(e);
        }
        if (!r.all_of<C>(e)) return;
        r.patch<C>(e, [&](C &record) {
            auto *field = reinterpret_cast<std::byte *>(&record) + target.Offset;
            if (target.Kind == ValueKind::Bool) {
                const uint32_t flag = in[0] != 0.f;
                std::memcpy(field, &flag, sizeof flag);
            } else {
                std::memcpy(field, in.data(), bytes);
            }
        });
        if constexpr (std::same_as<C, Visibility>) ApplyVisibility(r, e);
    });
}

bool SameValue(const ChannelTarget &target, std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) return false;
    // Opposite quaternions rotate the same.
    float sign = 1.f;
    if (target.Kind == ValueKind::Quaternion && a.size() == 4 && a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] < 0.f) sign = -1.f;
    for (size_t i = 0; i < a.size(); ++i)
        if (!NearlyEqual(a[i], sign * b[i])) return false;
    return true;
}
} // namespace animation
