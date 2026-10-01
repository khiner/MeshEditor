#include "mesh/TopologyReadView.h"
#include "Profile.h"

#include "mesh/MeshClosure.h"
#include "mesh/MeshStore.h"
#include "mesh/PageFootprint.h"
#include "state/Scene.h"

TopologyReadView::TopologyReadView(state::Scene &r, uint32_t id, const MeshClosure &neighborhood, const BufferArena<uint32_t> &storage) {
    const profile::CpuScope scope{"TopologyReadView"};
    if (!neighborhood.Counts[0]) return;
    auto &meshes = r.Context.get<MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    Connectivity = meshes.GetConnectivityRef(id);
    const auto vertex_origin = a.Vertices.First(record.Vertices);
    const auto vertices = WorkBlocks(storage, neighborhood.Elements[0], neighborhood.Counts[0], vertex_origin);
    const auto halfedges = WorkBlocks(storage, neighborhood.Elements[1], neighborhood.Counts[1]);
    const auto faces = WorkBlocks(storage, neighborhood.Elements[2], neighborhood.Counts[2]);
    const auto edges = WorkBlocks(storage, neighborhood.Elements[3], neighborhood.Counts[3]);
    // Face loops name the neighboring corners and the ring vertices that rules read around each halfedge.
    // Line corners belong to no loop, so a neighborhood without faces names them and their vertices directly.
    std::vector<uint32_t> corners, ring = vertices, fans, roots;
    {
        const auto ranges = a.FaceRanges.Buffer.GetSpan<uvec2>();
        const auto corner_vertices = a.FaceCorners.Buffer.GetSpan<uint32_t>();
        const auto add = [&](uint32_t h) {
            AddBlock(corners, h);
            AddBlock(ring, corner_vertices[h]);
        };
        ForEachWorkHandle(storage, neighborhood.Elements[2], neighborhood.Counts[2], 0u, [&](uint32_t face) {
            for (auto h = ranges[face].x; h < ranges[face].y; ++h) add(h);
        });
        if (!neighborhood.Counts[2]) ForEachWorkHandle(storage, neighborhood.Elements[1], neighborhood.Counts[1], 0u, add);
    }
    {
        const auto incoming = a.VertexCorners.Buffer.GetSpan<uvec2>();
        ForEachWorkHandle(storage, neighborhood.Elements[0], neighborhood.Counts[0], vertex_origin, [&](uint32_t v) {
            if (v >= incoming.size()) return;
            const auto fan = incoming[v];
            for (auto item = fan.x; item < fan.x + fan.y; item = (item / MeshElementBlockSize + 1u) * MeshElementBlockSize) AddBlock(fans, item);
        });
    }
    {
        // A sector root can be a corner outside the neighborhood.
        const auto tables = a.CornerSectors.Blocks.Buffer.GetSpan<uint32_t>();
        const auto values = a.CornerSectors.Values.Buffer.GetSpan<uint32_t>();
        ForEachWorkHandle(storage, neighborhood.Elements[1], neighborhood.Counts[1], 0u, [&](uint32_t h) {
            const auto block = h / MeshElementBlockSize;
            if (block >= tables.size() || !tables[block]) return;
            const auto root = values[(tables[block] - 1u) * MeshElementBlockSize + h % MeshElementBlockSize];
            if (root != InvalidOffset) AddBlock(roots, root);
        });
    }
    PageFootprint pages;
    pages.Add(a.Vertices.Buffer, ring, BlockBytes<Vertex>);
    pages.Add(a.BaseVertexNormals.Buffer, ring, BlockBytes<vec3>);
    pages.Add(a.OutgoingHalfedges.Buffer, vertices, BlockBytes<uint32_t>);
    pages.Add(a.VertexCorners.Buffer, vertices, BlockBytes<uvec2>);
    pages.Add(a.VertexFans.Items.Buffer, fans, BlockBytes<uvec2>);
    pages.Add(a.VertexSelection.Buffer, vertices, sizeof(MeshArenas::SelectionBlock));
    if (record.VertexAttributes & MeshAttributeBit_Color0) pages.Attribute(a.VertexColors, vertices);
    if (record.SkinBlocksReady) pages.Attribute(a.Skin, vertices);
    if (record.MorphBlocksReady) pages.Attribute(a.Morph, vertices, record.MorphTargetCount);
    pages.Add(a.FaceCorners.Buffer, corners, BlockBytes<uint32_t>);
    for (const auto *buffer : {&a.OppositeHalfedges.Buffer, &a.HalfedgeEdges.Buffer, &a.HalfedgeFaces.Buffer})
        pages.Add(*buffer, halfedges, BlockBytes<uint32_t>);
    if (record.CornerAttributes & MeshAttributeBit_Tangent) pages.Attribute(a.CornerTangents, halfedges);
    if (record.CornerAttributes & MeshAttributeBit_Color0) pages.Attribute(a.CornerColors, halfedges);
    for (uint32_t uv = 0u; uv < 4u; ++uv)
        if (record.CornerAttributes & (MeshAttributeBit_TexCoord0 << uv)) pages.Attribute(a.CornerUvs[uv], halfedges);
    pages.Attribute(a.CustomNormals, halfedges);
    pages.Attribute(a.CornerSectors, halfedges);
    pages.Attribute(a.NormalSectors, roots);
    pages.Add(a.FaceTriangles.Buffer, faces, BlockBytes<uint32_t>);
    pages.Add(a.FaceRanges.Buffer, faces, BlockBytes<uvec2>);
    pages.Add(a.FaceSharpness.Buffer, faces, BlockBytes<uint8_t>);
    pages.Add(a.BaseFaceNormals.Buffer, faces, BlockBytes<vec3>);
    pages.Add(a.FaceSelection.Buffer, faces, sizeof(MeshArenas::SelectionBlock));
    pages.Attribute(a.FacePrimitives, faces);
    pages.Add(a.EdgeHalfedges.Buffer, edges, BlockBytes<uint32_t>);
    pages.Add(a.EdgeSharpness.Buffer, edges, BlockBytes<uint8_t>);
    pages.Add(a.EdgeSelection.Buffer, edges, sizeof(MeshArenas::SelectionBlock));

    // Each binding reads its clone at canonical offsets.
    std::vector<std::pair<const mtl::Buffer *, uint32_t *>> bindings{
        {&a.Vertices.Buffer, &Arenas.VertexSlot}, {&a.FaceCorners.Buffer, &Arenas.CornerSlot},
        {&a.FaceTriangles.Buffer, &Arenas.FaceTriangleStartSlot}, {&a.FaceSharpness.Buffer, &Arenas.FaceSharpnessSlot},
        {&a.EdgeSharpness.Buffer, &Arenas.EdgeSharpnessSlot},
        {&a.BaseVertexNormals.Buffer, &Arenas.BaseVertexNormalSlot}, {&a.BaseFaceNormals.Buffer, &Arenas.BaseFaceNormalSlot},
        {&a.OutgoingHalfedges.Buffer, &Connectivity.Outgoing.Slot}, {&a.OppositeHalfedges.Buffer, &Connectivity.Opposites.Slot},
        {&a.HalfedgeEdges.Buffer, &Connectivity.HalfedgeEdges.Slot}, {&a.HalfedgeFaces.Buffer, &Connectivity.HalfedgeFaces.Slot},
        {&a.FaceRanges.Buffer, &Connectivity.FaceRanges.Slot}, {&a.EdgeHalfedges.Buffer, &Connectivity.Edges.Slot},
        {&a.VertexCorners.Buffer, &Connectivity.VertexCorners.Slot}, {&a.VertexFans.Items.Buffer, &Connectivity.FanItemsSlot},
        {&a.VertexSelection.Buffer, &Selection[0].Slot}, {&a.EdgeSelection.Buffer, &Selection[1].Slot},
        {&a.FaceSelection.Buffer, &Selection[2].Slot},
    };
    const auto attribute = [&](const auto &source, ElementAttributeRef &ref) {
        bindings.push_back({&source.Blocks.Buffer, &ref.BlocksSlot});
        bindings.push_back({&source.Values.Buffer, &ref.ValuesSlot});
    };
    attribute(a.FacePrimitives, Arenas.FacePrimitives);
    attribute(a.Skin, Arenas.Skin);
    attribute(a.Morph, Arenas.Morph);
    attribute(a.CornerTangents, Arenas.CornerTangent);
    attribute(a.CornerColors, Arenas.CornerColor);
    attribute(a.VertexColors, Arenas.VertexColor);
    for (uint32_t uv = 0u; uv < 4u; ++uv) attribute(a.CornerUvs[uv], Arenas.CornerUvs[uv]);
    attribute(a.CustomNormals, Arenas.CustomNormals);
    attribute(a.CornerSectors, Arenas.CornerSectors);
    attribute(a.NormalSectors, Arenas.NormalSectors);
    std::vector<mtl::BufferFootprint> footprints;
    std::vector<uint32_t *> slots;
    for (const auto &[buffer, slot] : bindings) {
        *slot = InvalidSlot;
        if (const auto read = pages.Pages(*buffer); !read.empty()) {
            footprints.push_back({buffer, read});
            slots.push_back(slot);
        }
    }
    Clones = mtl::CloneFootprints(meshes.BufferContext(), footprints);
    for (size_t i = 0u; i < slots.size(); ++i) *slots[i] = Clones[i].Slot;
}
