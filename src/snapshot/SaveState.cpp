#include "snapshot/SaveState.h"
#include "state/Allocation.h"

#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "render/GpuBuffers.h"
#include "render/MaterialComponents.h"
#include "snapshot/SceneSnapshot.h"

#include "state/Scene.h"
#include <zpp_bits.h>

namespace snapshot {
namespace {
// Persist generations and free-list order for deterministic subsequent allocations.
std::vector<std::byte> SerializeEntities(const state::Scene &r) {
    std::vector<std::byte> bytes;
    const auto &allocation = r.AllocationState();
    zpp::bits::out{bytes}(allocation.Generations.View(), allocation.Free.View()).or_throw();
    return bytes;
}

// Persist the canonical GPU material array and parallel names because SourceAssets cannot reconstruct them.
std::vector<std::byte> SerializeMaterials(const state::Scene &r) {
    const auto &materials = r.ctx().get<const GpuBuffers>().Materials;
    const auto mapped = materials.Contents();
    const auto used = std::min(size_t(materials.UsedSize), mapped.size());
    const auto &names = r.ctx().get<const MaterialStore>().Names;

    std::vector<std::byte> out;
    zpp::bits::out archive{out};
    if (zpp::bits::failure(archive(mapped.first(used), names))) return {};
    out.resize(archive.position());
    return out;
}

void DeserializeMaterials(state::Scene &r, std::span<const std::byte> bytes) {
    std::vector<std::byte> material_bytes;
    std::vector<std::string> names;
    zpp::bits::in archive{bytes};
    if (zpp::bits::failure(archive(material_bytes, names))) return;

    auto &buffer = r.ctx().get<GpuBuffers>().Materials;
    buffer.Reserve(material_bytes.size());
    if (!material_bytes.empty()) buffer.Update(material_bytes, 0);
    buffer.UsedSize = material_bytes.size();
    r.ctx().get<MaterialStore>().Names = std::move(names);
}

void AppendLengthPrefixed(std::vector<std::byte> &out, std::span<const std::byte> section) {
    const uint64_t len = section.size();
    const auto *len_bytes = reinterpret_cast<const std::byte *>(&len);
    out.insert(out.end(), len_bytes, len_bytes + sizeof(len));
    out.append_range(section);
}

// Returns and consumes the next length-prefixed section, or returns empty without consuming truncated input.
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

std::vector<std::byte> SaveState(const state::Scene &r) {
    const auto entities = SerializeEntities(r);
    const auto scene = SnapshotSceneState(r);
    const auto materials = SerializeMaterials(r);
    const auto mesh = r.ctx().get<const MeshStore>().Serialize();

    std::vector<std::byte> out;
    out.reserve(3 * sizeof(uint64_t) + entities.size() + scene.size() + materials.size() + mesh.size());
    AppendLengthPrefixed(out, entities);
    AppendLengthPrefixed(out, scene);
    AppendLengthPrefixed(out, materials);
    out.append_range(mesh);
    return out;
}

void LoadState(state::Scene &r, std::span<const std::byte> bytes) {
    const auto entities = TakeLengthPrefixed(bytes);
    const auto scene = TakeLengthPrefixed(bytes);
    const auto materials = TakeLengthPrefixed(bytes);

    {
        std::vector<uint32_t> table, free;
        zpp::bits::in{entities}(table, free).or_throw();
        r.ResetEntities();
        for (auto v : table) r.AllocationState().Generations.PushBack(v);
        for (auto v : free) r.AllocationState().Free.PushBack(v);
        r.RebuildLiving();
    }

    // Restore MeshStore offsets before components that reference them.
    auto &meshes = r.ctx().get<MeshStore>();
    meshes.Deserialize(bytes);
    DeserializeMaterials(r, materials);
    RestoreSceneState(r, scene);

    // Rebuild derived arenas from restored connectivity and sharpness.
    std::vector<Mesh> restored;
    for (const auto e : r.view<const MeshHandle>()) restored.emplace_back(GetMesh(r, e));
    meshes.RebuildDerived(restored);
}
} // namespace snapshot
