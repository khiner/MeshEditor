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
    const auto vertex_table = a.Vertices.Entries(record.Vertices), corner_table = a.FaceCorners.Entries(record.FaceCorners);
    const auto face_table = a.FaceTriangles.Entries(record.FaceData), edge_table = a.EdgeHalfedges.Entries(record.EdgeData);
    const auto vertices = WorkBlocks(storage, neighborhood.Elements[0], neighborhood.Counts[0], vertex_table);
    const auto halfedges = WorkBlocks(storage, neighborhood.Elements[1], neighborhood.Counts[1], corner_table);
    const auto faces = WorkBlocks(storage, neighborhood.Elements[2], neighborhood.Counts[2], face_table);
    const auto edges = WorkBlocks(storage, neighborhood.Elements[3], neighborhood.Counts[3], edge_table);
    // Face loops name the neighboring corners and the ring vertices that rules read around each halfedge.
    // Wire corners belong to no loop, so the neighborhood also names its corners directly.
    std::vector<uint32_t> corners, ring = vertices, fans, roots;
    {
        const auto ranges = a.FaceRanges.Buffer.GetSpan<uvec2>();
        const auto corner_vertices = a.FaceCorners.Buffer.GetSpan<uint32_t>();
        const auto add = [&](uint32_t h) {
            const auto address = Resolve<MeshElementBlockSize>(corner_table, ElementHandle{h}).Value();
            AddBlock(corners, address);
            AddBlock(ring, Resolve<MeshElementBlockSize>(vertex_table, ElementHandle{corner_vertices[address]}).Value());
        };
        ForEachWorkHandle(storage, neighborhood.Elements[2], neighborhood.Counts[2], 0u, [&](uint32_t face) {
            const auto range = ranges[Resolve<MeshElementBlockSize>(face_table, ElementHandle{face}).Value()];
            for (auto h = range.x; h < range.y; ++h) add(h);
        });
        ForEachWorkHandle(storage, neighborhood.Elements[1], neighborhood.Counts[1], 0u, add);
    }
    {
        const auto incoming = a.VertexCorners.Buffer.GetSpan<uvec2>();
        ForEachWorkHandle(storage, neighborhood.Elements[0], neighborhood.Counts[0], 0u, [&](uint32_t v) {
            const auto fan = incoming[Resolve<MeshElementBlockSize>(vertex_table, ElementHandle{v}).Value()];
            for (auto item = fan.x; item < fan.x + fan.y; item = (item / MeshElementBlockSize + 1u) * MeshElementBlockSize) AddBlock(fans, item);
        });
    }
    if (const auto sectors = a.CornerSectors.View(record.CornerSectors); !sectors.empty()) {
        // A sector root can be a corner outside the neighborhood.
        ForEachWorkHandle(storage, neighborhood.Elements[1], neighborhood.Counts[1], 0u, [&](uint32_t h) {
            if (const auto root = sectors[h]; root != InvalidOffset) AddBlock(roots, Resolve<MeshElementBlockSize>(corner_table, ElementHandle{root}).Value());
        });
    }
    const auto table = [&](const auto &arena, ElementSetRef set) {
        if (!set) return;
        const auto &header = arena.Set(set);
        std::vector<uint32_t> entries(header.BlockCount);
        std::ranges::iota(entries, header.First);
        Pages.Add(arena.Table.Buffer, entries, sizeof(uint32_t));
    };
    table(a.Vertices, record.Vertices);
    table(a.FaceCorners, record.FaceCorners);
    table(a.EdgeHalfedges, record.EdgeData);
    table(a.FaceTriangles, record.FaceData);
    table(a.Triangles, record.TriangleData);
    Pages.Add(a.Vertices.Buffer, ring, BlockBytes<Vertex>);
    Pages.Add(a.BaseVertexNormals.Buffer, ring, BlockBytes<vec3>);
    Pages.Add(a.OutgoingHalfedges.Buffer, vertices, BlockBytes<uint32_t>);
    Pages.Add(a.VertexCorners.Buffer, vertices, BlockBytes<uvec2>);
    Pages.Add(a.VertexFans.Items.Buffer, fans, BlockBytes<uvec2>);
    Pages.Add(a.VertexHidden.Buffer, vertices, sizeof(SelectionBlock));
    const auto vertex_members = a.Vertices.Blocks.Buffer.GetSpan<MeshElementBlock>(), corner_members = a.FaceCorners.Blocks.Buffer.GetSpan<MeshElementBlock>();
    Pages.Layer(a.VertexColors.Values, record.VertexColors, vertex_members, vertices);
    Pages.Layer(a.Skin.Values, record.Skin, vertex_members, vertices);
    Pages.Layer(a.Morph.Values, record.Morph, vertex_members, vertices, record.MorphTargetCount);
    Pages.Add(a.FaceCorners.Buffer, corners, BlockBytes<uint32_t>);
    for (const auto *buffer : {&a.OppositeHalfedges.Buffer, &a.HalfedgeEdges.Buffer, &a.HalfedgeFaces.Buffer}) {
        Pages.Add(*buffer, halfedges, BlockBytes<uint32_t>);
    }
    Pages.Layer(a.CornerTangents.Values, record.CornerTangents, corner_members, halfedges);
    Pages.Layer(a.CornerColors.Values, record.CornerColors, corner_members, halfedges);
    for (uint32_t uv = 0u; uv < 4u; ++uv) Pages.Layer(a.CornerUvs[uv].Values, record.CornerUvs[uv], corner_members, halfedges);
    Pages.Layer(a.CustomNormals.Values, record.CustomNormals, corner_members, halfedges);
    Pages.Layer(a.CornerSectors.Values, record.CornerSectors, corner_members, halfedges);
    Pages.Layer(a.NormalSectors.Values, record.NormalSectors, corner_members, roots);
    Pages.Add(a.FaceTriangles.Buffer, faces, BlockBytes<uint32_t>);
    Pages.Add(a.FaceRanges.Buffer, faces, BlockBytes<uvec2>);
    Pages.Add(a.FaceSharpness.Buffer, faces, BlockBytes<uint8_t>);
    Pages.Add(a.BaseFaceNormals.Buffer, faces, BlockBytes<vec3>);
    Pages.Add(a.FaceHidden.Buffer, faces, sizeof(SelectionBlock));
    Pages.Layer(a.FacePrimitives.Values, record.FacePrimitives, a.FaceTriangles.Blocks.Buffer.GetSpan<MeshElementBlock>(), faces);
    Pages.Layer(a.VertexPrimitives.Values, record.VertexPrimitives, vertex_members, vertices);
    Pages.Add(a.EdgeHalfedges.Buffer, edges, BlockBytes<uint32_t>);
    Pages.Add(a.EdgeSharpness.Buffer, edges, BlockBytes<uint8_t>);
    Pages.Add(a.EdgeHidden.Buffer, edges, sizeof(SelectionBlock));
    // Named attributes read their values through these clones at their ranges before the edit's insertions.
    const std::array domain_blocks{&vertices, &edges, &faces, &halfedges}; // By AttributeDomain
    meshes.ForEachNamedAttribute(id, [&](uint32_t, const NamedAttribute &entry) {
        const auto members = WithDomain(a, ElementDomainOf(entry.Domain), [](const auto &arena) { return arena.Blocks.Buffer.template GetSpan<MeshElementBlock>(); });
        WithAttributeValues(a.Named, entry.Type, [&](const auto &values) { Pages.Layer(values, entry.Values, members, *domain_blocks[uint32_t(entry.Domain)]); });
    });
}

void TopologyReadView::Clone(state::Scene &r) {
    const profile::CpuScope scope{"TopologyReadViewClone"};
    const auto &meshes = r.Context.get<MeshStore>();
    const auto &a = meshes.Arenas();
    // Each binding reads its clone at the source offsets.
    std::vector<std::pair<const mtl::Buffer *, uint32_t *>> bindings{
        {&a.Vertices.Buffer, &Arenas.VertexSlot},
        {&a.FaceCorners.Buffer, &Arenas.CornerSlot},
        {&a.FaceTriangles.Buffer, &Arenas.FaceTriangleStartSlot},
        {&a.FaceSharpness.Buffer, &Arenas.FaceSharpnessSlot},
        {&a.EdgeSharpness.Buffer, &Arenas.EdgeSharpnessSlot},
        {&a.BaseVertexNormals.Buffer, &Arenas.BaseVertexNormalSlot},
        {&a.BaseFaceNormals.Buffer, &Arenas.BaseFaceNormalSlot},
        {&a.OutgoingHalfedges.Buffer, &Connectivity.OutgoingSlot},
        {&a.OppositeHalfedges.Buffer, &Connectivity.OppositesSlot},
        {&a.HalfedgeEdges.Buffer, &Connectivity.HalfedgeEdgesSlot},
        {&a.HalfedgeFaces.Buffer, &Connectivity.HalfedgeFacesSlot},
        {&a.FaceRanges.Buffer, &Connectivity.FaceRangesSlot},
        {&a.EdgeHalfedges.Buffer, &Connectivity.EdgesSlot},
        {&a.VertexCorners.Buffer, &Connectivity.VertexCornersSlot},
        {&a.VertexFans.Items.Buffer, &Connectivity.FanItemsSlot},
        {&a.Vertices.Table.Buffer, &Connectivity.Tables.Vertices.Slot},
        {&a.FaceCorners.Table.Buffer, &Connectivity.Tables.Halfedges.Slot},
        {&a.EdgeHalfedges.Table.Buffer, &Connectivity.Tables.Edges.Slot},
        {&a.FaceTriangles.Table.Buffer, &Connectivity.Tables.Faces.Slot},
        {&a.Triangles.Table.Buffer, &Connectivity.Tables.Triangles.Slot},
        {&a.VertexHidden.Buffer, &Arenas.VertexHiddenSlot},
        {&a.EdgeHidden.Buffer, &Arenas.EdgeHiddenSlot},
        {&a.FaceHidden.Buffer, &Arenas.FaceHiddenSlot},
    };
    auto &Layers = Arenas.Layers;
    const auto layer = [&](const auto &source, SlotOffset &ref) { bindings.push_back({&source.Values.Buffer, &ref.Slot}); };
    layer(a.FacePrimitives, Layers.FacePrimitives);
    layer(a.VertexPrimitives, Layers.VertexPrimitives);
    layer(a.Skin, Layers.Skin);
    layer(a.Morph, Layers.Morph);
    layer(a.CornerTangents, Layers.CornerTangent);
    layer(a.CornerColors, Layers.CornerColor);
    layer(a.VertexColors, Layers.VertexColor);
    for (uint32_t uv = 0u; uv < 4u; ++uv) layer(a.CornerUvs[uv], Layers.CornerUvs[uv]);
    layer(a.CustomNormals, Layers.CustomNormals);
    layer(a.CornerSectors, Layers.CornerSectors);
    layer(a.NormalSectors, Layers.NormalSectors);
    for (const auto type : AttributeTypes) WithAttributeValues(a.Named, type, [&](const auto &values) { bindings.push_back({&values.Buffer, &AttributeValues[uint32_t(type)]}); });
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
    const auto tables = meshes.GetElementTables(id);
    auto source = Connectivity;
    source.Tables.Vertices.Offset = tables.Vertices.Offset;
    source.Tables.Halfedges.Offset = tables.Halfedges.Offset;
    source.Tables.Edges.Offset = tables.Edges.Offset;
    source.Tables.Faces.Offset = tables.Faces.Offset;
    source.Tables.Triangles.Offset = tables.Triangles.Offset;
    return source;
}

ElementLayers TopologyReadView::SourceLayers(const MeshStore &meshes, uint32_t id) const {
    auto layers = meshes.Layers(id);
    const auto &Layers = Arenas.Layers;
    // Read present layers from clones at source offsets; clone every page the edit reads.
    const auto at = [](SlotOffset &ref, SlotOffset clone) { ref.Slot = ref.Slot == InvalidSlot ? InvalidSlot : clone.Slot; };
    at(layers.FacePrimitives, Layers.FacePrimitives);
    at(layers.VertexPrimitives, Layers.VertexPrimitives);
    at(layers.Skin, Layers.Skin);
    at(layers.Morph, Layers.Morph);
    at(layers.CornerTangent, Layers.CornerTangent);
    at(layers.CornerColor, Layers.CornerColor);
    at(layers.VertexColor, Layers.VertexColor);
    for (uint32_t uv = 0u; uv < 4u; ++uv) at(layers.CornerUvs[uv], Layers.CornerUvs[uv]);
    at(layers.CustomNormals, Layers.CustomNormals);
    at(layers.CornerSectors, Layers.CornerSectors);
    at(layers.NormalSectors, Layers.NormalSectors);
    return layers;
}
