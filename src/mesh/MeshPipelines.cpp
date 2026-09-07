#include "mesh/MeshPipelines.h"

MeshPipelines::MeshPipelines(mtl::LibraryCache &libraries)
    : VertexAdjacency{libraries}, VertexWeld{libraries}, MeshConnectivity{libraries} {}

void MeshPipelines::CompileShaders(mtl::LibraryCache &libraries) {
    for (auto *pipeline : {&VertexAdjacency.Zero, &VertexAdjacency.Count, &VertexAdjacency.BlockSum, &VertexAdjacency.BlockPrefix, &VertexAdjacency.Offsets, &VertexAdjacency.Scatter, &VertexAdjacency.Sort, &VertexWeld.TableInit, &VertexWeld.Insert, &VertexWeld.MarkReps, &VertexWeld.BlockSum, &VertexWeld.BlockPrefix, &VertexWeld.Scan, &VertexWeld.Emit, &VertexWeld.Compact, &VertexWeld.WriteBack, &VertexWeld.RemapCorners, &MeshConnectivity.Zero, &MeshConnectivity.Count, &MeshConnectivity.BlockSum, &MeshConnectivity.BlockPrefix, &MeshConnectivity.Offsets, &MeshConnectivity.Scatter, &MeshConnectivity.Pair, &MeshConnectivity.Bits, &MeshConnectivity.WordBlockSum, &MeshConnectivity.WordBlockPrefix, &MeshConnectivity.Ranks, &MeshConnectivity.Samples}) pipeline->Compile(libraries);
}
