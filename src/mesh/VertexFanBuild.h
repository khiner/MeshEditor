#pragma once
#include <cstdint>
#include <span>

namespace mtl { struct ComputeChain; }

struct MeshConnectivityJob;
namespace state { struct Scene; }

// Records the rebuild of the packed fans of connectivity jobs' writable vertices into `chain`.
// Each job's fans fill one run of its supplied incidence count, allocated before recording, and a device scan places each fan inside it.
// Work includes complete incidence of writable vertices. Other vertices retain their roots and items.
// A fresh job has dense vertices and corners and no former runs.
// Record after canonical corner/face ownership is available.
// The recording captures the runs and the roots in `vertex_blocks`, the ascending blocks of every writable vertex.
// The chain's next submit frees the writable vertices' former runs and each run's unused tail.
void EncodeVertexFans(state::Scene &, mtl::ComputeChain &, std::span<const MeshConnectivityJob>, std::span<const uint32_t> vertex_blocks, bool fresh = false);
