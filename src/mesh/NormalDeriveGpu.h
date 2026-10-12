#pragma once

#include "gpu/ElementWork.h"
#include "gpu/NormalDeriveEntry.h"
#include "gpu/NormalDerivePushConstants.h"
#include "metal/BufferArena.h"
#include "state/Entity.h"

struct MeshStore;
namespace mtl {
struct ComputeChain;
}

// Captures the base normal pages the work of record `id`'s entry writes.
// These are its vertex and face normals and the sector normals around its vertices.
void CaptureNormalWrites(state::Scene &, uint32_t id, const NormalDeriveEntry &, const BufferArena<uint32_t> &vertex_work, const BufferArena<uint32_t> &face_work);

// The record's geometry and connectivity.
// Callers choose base or posed destinations.
std::optional<NormalDeriveEntry> MakeDeriveEntryInputs(const MeshStore &, uint32_t store_id);

// Records the same two-phase kernel as frame and sparse edit derivation, with each entry from the record of `ids` at its index.
// Only job metadata is uploaded, and geometry and normal values stay in their GPU arenas.
void EncodeDeriveNormals(state::Scene &, mtl::ComputeChain &, std::span<const uint32_t> ids, std::span<const NormalDeriveEntry>, NormalDerivePushConstants);
// Records the base normals of every vertex and face of the meshes.
// It submits the chain once to gather their membership, and the derive runs with the chain's next submit.
void EncodeDeriveAllNormals(state::Scene &, mtl::ComputeChain &, std::span<const uint32_t> store_ids);
// One mesh's local normal change.
// Faces cover the changed face normals, and vertices cover every affected normal fan.
struct LocalNormalWork {
    uint32_t StoreId;
    ElementWork Vertices;
    uint32_t VertexCount;
    ElementWork Faces;
    uint32_t FaceCount;
};
// Records base normals after local changes to the meshes, in one derive.
// Every change's vertices and faces are finished handle membership in `work`.
void EncodeDeriveMeshNormals(state::Scene &, mtl::ComputeChain &, const BufferArena<uint32_t> &work, std::span<const LocalNormalWork>);
