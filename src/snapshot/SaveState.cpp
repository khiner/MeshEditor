#include "snapshot/SaveState.h"

#include "mesh/MeshStore.h"
#include "render/GpuBuffers.h"
#include "render/MaterialComponents.h"
#include "selection/SelectionBitset.h"
#include "snapshot/SceneSnapshot.h"

#include <entt/entity/registry.hpp>
#include <zpp_bits.h>

namespace snapshot {
namespace {
// The PBRMaterial GPU array (GpuBuffers::Materials) is canonical material state with no per-entity
// Persistent backing, and is not re-derivable from SourceAssets. Persist it wholesale, like the MeshStore
// arenas, so every primitive's MeshMaterialAssignment index and every baked bindless texture slot stays
// valid on restore. MaterialStore::Names rides along (the parallel CPU name list).
std::vector<std::byte> SerializeMaterials(const entt::registry &r) {
    const auto &materials = r.ctx().get<const GpuBuffers>().Materials;
    const auto mapped = materials.Buffer.GetMappedData();
    const auto used = std::min(size_t(materials.Buffer.UsedSize), mapped.size());
    std::vector<std::byte> material_bytes{mapped.begin(), mapped.begin() + used};
    auto names = r.ctx().get<const MaterialStore>().Names;

    std::vector<std::byte> out;
    zpp::bits::out archive{out};
    if (zpp::bits::failure(archive(material_bytes, names))) return {};
    out.resize(archive.position());
    return out;
}

void DeserializeMaterials(entt::registry &r, std::span<const std::byte> bytes) {
    std::vector<std::byte> material_bytes;
    std::vector<std::string> names;
    zpp::bits::in archive{bytes};
    if (zpp::bits::failure(archive(material_bytes, names))) return;

    auto &buffer = r.ctx().get<GpuBuffers>().Materials.Buffer;
    buffer.Reserve(material_bytes.size());
    if (!material_bytes.empty()) buffer.Update(material_bytes, 0);
    buffer.UsedSize = material_bytes.size();
    r.ctx().get<MaterialStore>().Names = std::move(names);
}

// Element-selection bits, up to the end of the last mesh's bitset range. The ranges themselves are
// Persistent components, so restoring both brings back the full edit-mode element selection.
std::vector<std::byte> SerializeSelectionBits(const entt::registry &r) {
    uint32_t max_end = 0;
    for (const auto [_, br] : r.view<const MeshSelectionBitsetRange>().each()) max_end = std::max(max_end, br.Offset + br.Count);
    const auto mapped = r.ctx().get<const GpuBuffers>().SelectionBitset.Buffer.GetMappedData();
    const auto used = std::min(size_t((max_end + 31) / 32) * sizeof(uint32_t), mapped.size());
    return {mapped.begin(), mapped.begin() + used};
}

void DeserializeSelectionBits(entt::registry &r, std::span<const std::byte> bytes) {
    auto &buffer = r.ctx().get<GpuBuffers>().SelectionBitset.Buffer;
    const auto mapped = buffer.GetMutableRange(0, buffer.GetMappedData().size());
    std::memset(mapped.data(), 0, mapped.size());
    if (!bytes.empty()) buffer.Update(bytes, 0);
}

void AppendLengthPrefixed(std::vector<std::byte> &out, std::span<const std::byte> section) {
    const uint64_t len = section.size();
    const auto *len_bytes = reinterpret_cast<const std::byte *>(&len);
    out.insert(out.end(), len_bytes, len_bytes + sizeof(len));
    out.append_range(section);
}

// Reads a length-prefixed section from the front of `bytes`, advancing it past the section. Returns the
// section bytes, or empty + an unadvanced `bytes` on truncation.
std::span<const std::byte> TakeLengthPrefixed(std::span<const std::byte> &bytes) {
    if (bytes.size() < sizeof(uint64_t)) return {};
    uint64_t len;
    std::memcpy(&len, bytes.data(), sizeof(len));
    const auto rest = bytes.subspan(sizeof(len));
    if (len > rest.size()) return {};
    bytes = rest.subspan(len);
    return rest.subspan(0, len);
}
} // namespace

std::vector<std::byte> SaveState(const entt::registry &r) {
    const auto scene = SnapshotSceneState(r);
    const auto materials = SerializeMaterials(r);
    const auto selection_bits = SerializeSelectionBits(r);
    const auto mesh = r.ctx().get<const MeshStore>().Serialize();

    std::vector<std::byte> out;
    out.reserve(3 * sizeof(uint64_t) + scene.size() + materials.size() + selection_bits.size() + mesh.size());
    AppendLengthPrefixed(out, scene);
    AppendLengthPrefixed(out, materials);
    AppendLengthPrefixed(out, selection_bits);
    out.append_range(mesh); // trailing section, no length prefix
    return out;
}

void LoadState(entt::registry &r, std::span<const std::byte> bytes) {
    const auto scene = TakeLengthPrefixed(bytes);
    const auto materials = TakeLengthPrefixed(bytes);
    const auto selection_bits = TakeLengthPrefixed(bytes);

    // MeshStore first: restoring its arenas and entries keeps every Range/StoreId offset valid before the components that reference them are restored.
    r.ctx().get<MeshStore>().Deserialize(bytes);
    DeserializeMaterials(r, materials);
    DeserializeSelectionBits(r, selection_bits);
    RestoreSceneState(r, scene);
}
} // namespace snapshot
