#include "mesh/TopologyReadView.h"
#include "Profile.h"

#include "mesh/MeshClosure.h"
#include "mesh/MeshStore.h"
#include "state/Scene.h"

void TopologyReadView::Add(state::Scene &r, uint32_t id, const MeshClosure &neighborhood, const BufferArena<uint32_t> &storage) {
    const profile::CpuScope scope{"TopologyReadView"};
    if (!neighborhood.Counts[0]) return;
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    const auto vertex_origin = a.Vertices.First(record.Vertices);
    const auto vertices = WorkBlocks(storage, neighborhood.Elements[0], neighborhood.Counts[0], vertex_origin);
    const auto halfedges = WorkBlocks(storage, neighborhood.Elements[1], neighborhood.Counts[1]);
    const auto faces = WorkBlocks(storage, neighborhood.Elements[2], neighborhood.Counts[2]);
    const auto edges = WorkBlocks(storage, neighborhood.Elements[3], neighborhood.Counts[3]);
    // Face loops name the neighboring corners and the ring vertices that rules read around each halfedge.
    // Wire corners belong to no loop, so the neighborhood also names its corners directly.
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
        ForEachWorkHandle(storage, neighborhood.Elements[1], neighborhood.Counts[1], 0u, add);
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
    Pages.Add(a.Vertices.Buffer, ring, BlockBytes<Vertex>);
    Pages.Add(a.BaseVertexNormals.Buffer, ring, BlockBytes<vec3>);
    Pages.Add(a.OutgoingHalfedges.Buffer, vertices, BlockBytes<uint32_t>);
    Pages.Add(a.VertexCorners.Buffer, vertices, BlockBytes<uvec2>);
    Pages.Add(a.VertexFans.Items.Buffer, fans, BlockBytes<uvec2>);
    Pages.Add(a.VertexSelection.Buffer, vertices, sizeof(MeshArenas::SelectionBlock));
    Pages.Add(a.VertexHidden.Buffer, vertices, sizeof(MeshArenas::SelectionBlock));
    if (record.VertexAttributes & MeshAttributeBit_Color0) Pages.Attribute(a.VertexColors, vertices);
    if (record.SkinBlocksReady) Pages.Attribute(a.Skin, vertices);
    if (record.MorphBlocksReady) Pages.Attribute(a.Morph, vertices, record.MorphTargetCount);
    Pages.Add(a.FaceCorners.Buffer, corners, BlockBytes<uint32_t>);
    for (const auto *buffer : {&a.OppositeHalfedges.Buffer, &a.HalfedgeEdges.Buffer, &a.HalfedgeFaces.Buffer})
        Pages.Add(*buffer, halfedges, BlockBytes<uint32_t>);
    if (record.CornerAttributes & MeshAttributeBit_Tangent) Pages.Attribute(a.CornerTangents, halfedges);
    if (record.CornerAttributes & MeshAttributeBit_Color0) Pages.Attribute(a.CornerColors, halfedges);
    for (uint32_t uv = 0u; uv < 4u; ++uv)
        if (record.CornerAttributes & (MeshAttributeBit_TexCoord0 << uv)) Pages.Attribute(a.CornerUvs[uv], halfedges);
    Pages.Attribute(a.CustomNormals, halfedges);
    Pages.Attribute(a.CornerSectors, halfedges);
    Pages.Attribute(a.NormalSectors, roots);
    Pages.Add(a.FaceTriangles.Buffer, faces, BlockBytes<uint32_t>);
    Pages.Add(a.FaceRanges.Buffer, faces, BlockBytes<uvec2>);
    Pages.Add(a.FaceSharpness.Buffer, faces, BlockBytes<uint8_t>);
    Pages.Add(a.BaseFaceNormals.Buffer, faces, BlockBytes<vec3>);
    Pages.Add(a.FaceSelection.Buffer, faces, sizeof(MeshArenas::SelectionBlock));
    Pages.Add(a.FaceHidden.Buffer, faces, sizeof(MeshArenas::SelectionBlock));
    Pages.Attribute(a.FacePrimitives, faces);
    if (record.VertexPrimitivesReady) Pages.Attribute(a.VertexPrimitives, vertices);
    Pages.Add(a.EdgeHalfedges.Buffer, edges, BlockBytes<uint32_t>);
    Pages.Add(a.EdgeSharpness.Buffer, edges, BlockBytes<uint8_t>);
    Pages.Add(a.EdgeSelection.Buffer, edges, sizeof(MeshArenas::SelectionBlock));
    Pages.Add(a.EdgeHidden.Buffer, edges, sizeof(MeshArenas::SelectionBlock));
}

void TopologyReadView::Clone(state::Scene &r) {
    const profile::CpuScope scope{"TopologyReadViewClone"};
    auto &meshes = r.Context.get<MeshStore>();
    const auto &a = meshes.Arenas();
    // Each binding reads its clone at canonical offsets.
    std::vector<std::pair<const mtl::Buffer *, uint32_t *>> bindings{
        {&a.Vertices.Buffer, &Arenas.VertexSlot},
        {&a.FaceCorners.Buffer, &Arenas.CornerSlot},
        {&a.FaceTriangles.Buffer, &Arenas.FaceTriangleStartSlot},
        {&a.FaceSharpness.Buffer, &Arenas.FaceSharpnessSlot},
        {&a.EdgeSharpness.Buffer, &Arenas.EdgeSharpnessSlot},
        {&a.BaseVertexNormals.Buffer, &Arenas.BaseVertexNormalSlot},
        {&a.BaseFaceNormals.Buffer, &Arenas.BaseFaceNormalSlot},
        {&a.OutgoingHalfedges.Buffer, &Connectivity.Outgoing.Slot},
        {&a.OppositeHalfedges.Buffer, &Connectivity.Opposites.Slot},
        {&a.HalfedgeEdges.Buffer, &Connectivity.HalfedgeEdges.Slot},
        {&a.HalfedgeFaces.Buffer, &Connectivity.HalfedgeFaces.Slot},
        {&a.FaceRanges.Buffer, &Connectivity.FaceRanges.Slot},
        {&a.EdgeHalfedges.Buffer, &Connectivity.Edges.Slot},
        {&a.VertexCorners.Buffer, &Connectivity.VertexCorners.Slot},
        {&a.VertexFans.Items.Buffer, &Connectivity.FanItemsSlot},
        {&a.VertexSelection.Buffer, &Selection[0].Slot},
        {&a.EdgeSelection.Buffer, &Selection[1].Slot},
        {&a.FaceSelection.Buffer, &Selection[2].Slot},
        {&a.VertexHidden.Buffer, &Arenas.VertexHiddenSlot},
        {&a.EdgeHidden.Buffer, &Arenas.EdgeHiddenSlot},
        {&a.FaceHidden.Buffer, &Arenas.FaceHiddenSlot},
    };
    const auto attribute = [&](const auto &source, ElementAttributeRef &ref) {
        bindings.push_back({&source.Blocks.Buffer, &ref.BlocksSlot});
        bindings.push_back({&source.Values.Buffer, &ref.ValuesSlot});
    };
    attribute(a.FacePrimitives, Arenas.FacePrimitives);
    attribute(a.VertexPrimitives, Arenas.VertexPrimitives);
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
        if (const auto read = Pages.Pages(*buffer); !read.empty()) {
            footprints.push_back({buffer, read});
            slots.push_back(slot);
        }
    }
    Clones = mtl::CloneFootprints(meshes.BufferContext(), footprints);
    for (size_t i = 0u; i < slots.size(); ++i) *slots[i] = Clones[i].Slot;
}

ConnectivityRef TopologyReadView::SourceConnectivity(const MeshStore &meshes, uint32_t id) const {
    const auto source = meshes.GetConnectivityRef(id);
    return {
        {Connectivity.Outgoing.Slot, source.Outgoing.Offset},
        {Connectivity.Opposites.Slot, source.Opposites.Offset},
        {Connectivity.HalfedgeEdges.Slot, source.HalfedgeEdges.Offset},
        {Connectivity.HalfedgeFaces.Slot, source.HalfedgeFaces.Offset},
        {Connectivity.FaceRanges.Slot, source.FaceRanges.Offset},
        {Connectivity.Edges.Slot, source.Edges.Offset},
        {Connectivity.VertexCorners.Slot, source.VertexCorners.Offset},
        Connectivity.FanItemsSlot,
    };
}
