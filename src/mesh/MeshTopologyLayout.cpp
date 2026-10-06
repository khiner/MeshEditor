#include "mesh/MeshTopologyLayout.h"
#include "mesh/ScratchChunks.h"

namespace {
uint32_t GpuCount(uint64_t count) {
    if (count > UINT32_MAX) throw std::length_error("Topology reservation exceeds the GPU's 32-bit element or scratch address space.");
    return uint32_t(count);
}
uint32_t Cuts(float value) {
    if (!std::isfinite(value) || double(value) > double(UINT32_MAX)) throw std::length_error("Topology cut count is outside the GPU address space.");
    return value < 1.f ? 1u : uint32_t(value);
}
} // namespace

MeshStore::TopologyCounts TopologyOutputBounds(const MeshTopologyTask &task, MeshStore::TopologyCounts source) {
    const uint64_t V = source.Vertices, H = source.Halfedges, F = source.Faces;
    const auto counts = [](uint64_t v, uint64_t h, uint64_t f) {
        return MeshStore::TopologyCounts{GpuCount(v), GpuCount(h), GpuCount(f)};
    };
    // Welding can split faces at repeated targets and leave loose edges.
    // Each face keeps at least three of the source corners.
    if (TopologyIsMerge(task.Op)) return counts(V, 2 * H, F ? H / 3u : 0u);
    switch (TopologyBaseOp(task.Op)) {
        case MeshTopologyOp::DeleteVertices:
        case MeshTopologyOp::DeleteLoose:
        case MeshTopologyOp::DeleteEdges:
        case MeshTopologyOp::DeleteOnlyEdgesFaces:
        case MeshTopologyOp::DeleteOnlyFaces: return counts(V, 2 * H, F);
        case MeshTopologyOp::ExtrudeVertices: return counts(2 * V, 2 * V, 0);
        case MeshTopologyOp::ExtrudeRegion: {
            const uint64_t steps = task.Op == MeshTopologyOp::ExtrudeRegion ? task.Steps : 1u;
            const auto vh = GpuCount(H * steps), vf = GpuCount(F * steps);
            return counts(V + GpuCount(V * steps), H + 5ull * vh + (task.Op == MeshTopologyOp::ExtrudeRegion ? 2u * V * steps : 0u), F + vf + vh);
        }
        case MeshTopologyOp::DuplicateGeometry:
        case MeshTopologyOp::SplitGeometry: return counts(2 * V, 6 * H, 2 * F + H);
        case MeshTopologyOp::ExtrudeEdges: return counts(2 * V, 5 * H, F + H);
        case MeshTopologyOp::ExtrudeFacesIndividual: return counts(V + H, 5 * H, F + H);
        case MeshTopologyOp::Wireframe: return counts(4 * V + H, 17 * H, F + 4 * H);
        case MeshTopologyOp::Subdivide: {
            const uint64_t c = (task.Flags & (TopologyFlagPlaneCuts | TopologyFlagListCuts | TopologyFlagScreenCuts)) ? 1u : Cuts(task.Param0);
            const uint64_t width = GpuCount(c + 1);
            const auto rows = GpuCount(width * width), inner = GpuCount(c * c);
            const auto face_rows = GpuCount(uint64_t(rows) * F), hc = GpuCount(H * c);
            return counts(V + hc + GpuCount(uint64_t(inner) * F), 4ull * face_rows + H + 2ull * hc, uint64_t(face_rows) + hc + F);
        }
        case MeshTopologyOp::Triangulate:
        case MeshTopologyOp::SplitNonplanarFaces:
        case MeshTopologyOp::SplitConcaveFaces:
        case MeshTopologyOp::Poke: return counts(V + F, 3 * H, H);
        case MeshTopologyOp::EdgeSplit: return counts(V + H, H, F);
        case MeshTopologyOp::AddPrimitives: {
            if (task.List.size() <= TopologyPrimitiveHeaderWords) throw std::invalid_argument("Topology primitive list has no header.");
            const uint64_t vertices = task.List[0], boundary = task.List[1], span = task.List[2];
            if (boundary ? boundary < 4u || boundary % 2u || !span || span >= boundary / 2u || vertices != (span - 1u) * (boundary / 2u - span - 1u) : vertices || span)
                throw std::invalid_argument("Topology primitive list has an invalid grid.");
            const uint64_t first_face = TopologyPrimitiveHeaderWords + boundary;
            if (first_face >= task.List.size()) throw std::invalid_argument("Topology primitive list has an incomplete boundary.");
            const uint64_t primitives = task.List[first_face];
            uint64_t cursor = first_face + 1, corners = 0, faces = 0;
            for (uint64_t f = 0; f < primitives; ++f) {
                if (cursor >= task.List.size()) throw std::invalid_argument("Topology face list has incomplete faces.");
                const uint64_t n = task.List[cursor++];
                if (n < 2 || n > (task.List.size() - cursor) / 2u) throw std::invalid_argument("Topology primitive list has an invalid loop.");
                cursor += 2u * n;
                corners += n;
                faces += n > 2u;
            }
            if (cursor != task.List.size()) throw std::invalid_argument("Topology face list has trailing words.");
            return counts(V + vertices, H + corners, F + faces);
        }
        case MeshTopologyOp::ReplaceFaces: {
            const auto &list = task.List;
            if (list.empty() || uint64_t(list[0]) + 1u >= list.size()) throw std::invalid_argument("Face replacement has an invalid vertex list.");
            const uint64_t offsets = 2ull + list[0], sources = list[offsets - 1u];
            if (sources > list.size() - offsets) throw std::invalid_argument("Face replacement has an invalid face list.");
            uint64_t cursor = offsets + sources, faces = 0u, corners = 0u;
            for (uint64_t f = 0u; f < sources; ++f) {
                if (list[offsets + f] != cursor || cursor + 2u > list.size()) throw std::invalid_argument("Face replacement has an invalid face offset.");
                ++cursor;
                const auto polygons = list[cursor++];
                faces += polygons;
                for (uint32_t i = 0u; i < polygons; ++i) {
                    if (cursor >= list.size()) throw std::invalid_argument("Face replacement has incomplete polygons.");
                    const uint64_t n = list[cursor++];
                    if (n < 3u || n > (list.size() - cursor) / 2u) throw std::invalid_argument("Face replacement has an invalid polygon.");
                    cursor += 2u * n;
                    corners += n;
                }
            }
            if (cursor != list.size()) throw std::invalid_argument("Face replacement has trailing words.");
            return counts(V, corners, faces);
        }
        case MeshTopologyOp::BevelVertices: {
            const uint64_t steps = std::max(task.Steps, 1u);
            const auto edge_steps = GpuCount(H * steps);
            const auto profile_edges = GpuCount(H * (steps - 1u));
            return counts(V + 2ull * edge_steps, 6ull * H + 8ull * profile_edges, F + V + 2ull * profile_edges);
        }
        case MeshTopologyOp::BevelEdges: {
            const uint64_t steps = std::max(task.Steps, 1u);
            const auto vertices_per_edge = GpuCount(2ull * steps + 1u);
            const auto edge_steps = GpuCount(H * steps);
            const auto edge_vertices = GpuCount(H * vertices_per_edge);
            return counts(V + edge_vertices, 6 * H + 4ull * edge_steps, F + edge_steps + V);
        }
        case MeshTopologyOp::ConnectVertices:
        case MeshTopologyOp::RotateEdges: return counts(V, 3 * H, H);
        default: return source;
    }
}

uint32_t TopologyTableWords(MeshTopologyOp op, MeshStore::TopologyCounts source) {
    const bool lines = TopologyJoinsLines(op);
    if (op != MeshTopologyOp::MergeByDistance && !lines) return 0u;
    // A mixed core can have one distinct edge per face corner.
    const uint64_t keys = std::max(op == MeshTopologyOp::MergeByDistance ? uint64_t(source.Vertices) : 0u, lines ? uint64_t(source.Halfedges) / (source.Faces ? 1u : 2u) : 0u);
    // Size the table for a load factor under three quarters.
    return GpuCount(std::bit_ceil(keys + keys / 2 + 1));
}

// Assigns the job's scratch runs from `base` and returns the words they span.
// Runs the operator does not use take no words.
uint32_t LayoutTopologyScratch(MeshTopologyJob &job, MeshStore::TopologyCounts source, MeshStore::TopologyCounts bounds, uint32_t base) {
    const auto op = job.Op;
    const uint64_t V = source.Vertices, H = source.Halfedges, F = source.Faces;
    uint64_t cursor = base;
    const auto take = [&](uint64_t words) {
        const auto first = GpuCount(cursor);
        cursor = GpuCount(cursor + words);
        return first;
    };
    // Counts cover every source vertex, halfedge, and face, then the appended-face list, then the scan terminator.
    job.CountEntries = GpuCount(V + H + F + 2);
    job.CountBlockCount = TileCount(job.CountEntries, BlockElements);
    const auto table = TopologyTableWords(op, source);
    job.TableMask = table > 0 ? table - 1 : 0u;
    job.StateOffset = take(2);
    job.SelectionOffset = take(job.SelectionElement != Element::None ? V + H + F : 0u);
    job.FlagVertexOffset = take(V);
    job.VertexTargetOffset = take(V);
    job.FlagHalfedgeOffset = take(H);
    job.FlagFaceOffset = take(F);
    job.CountsOffset = take(4ull * job.CountEntries);
    job.CountBlockOffset = take(4ull * job.CountBlockCount);
    job.VertexMapOffset = take(std::max(6ull * bounds.Vertices, bounds.Faces ? 4ull * bounds.Halfedges : 0ull));
    job.CornerMapOffset = take(8ull * bounds.Halfedges);
    job.FaceMapOffset = take(bounds.Faces);
    job.CornerProvenanceOffset = take(job.CornerAttributes & MeshAttributeBit_Normal ? 2ull * bounds.Halfedges : 0u);
    job.LabelOffset = take(TopologyIterates(op) ? 4 * F + 2 * V : 0u);
    // Triangulation keeps its per-face ear links and projected corners in
    // source-halfedge scratch, so polygon valence has no fixed shader limit.
    job.HalfedgeAuxOffset = take(op == MeshTopologyOp::ExtrudeRegion ? 2ull * H : op == MeshTopologyOp::Wireframe                                                                                                                 ? 2ull * H :
                                     op == MeshTopologyOp::Triangulate || op == MeshTopologyOp::SplitConcaveFaces                                                                                                                 ? 4ull * H :
                                     op == MeshTopologyOp::EdgeSplit || (TopologyMapsWireEdges(op) && op != MeshTopologyOp::Subdivide) || (job.Flags & (TopologyFlagListCuts | TopologyFlagListSelects | TopologyFlagScreenCuts)) ? H :
                                                                                                                                                                                                                                    0u);
    job.WireEdgeMapOffset = op == MeshTopologyOp::Subdivide || op == MeshTopologyOp::ExtrudeRegion ? take(H) : job.HalfedgeAuxOffset;
    const uint64_t split_width = op == MeshTopologyOp::Subdivide ?
        uint64_t((job.Flags & (TopologyFlagPlaneCuts | TopologyFlagListCuts | TopologyFlagScreenCuts)) ? 1u : Cuts(job.Param0)) + 1u :
        1u;
    job.FaceLoopOffset = take(TopologyIsDissolve(op) && TopologyJoinsLines(op) ? ((F ? H : 0ull) + (op == MeshTopologyOp::DissolveLimited ? 4ull * V : 0ull)) : TopologyIsMerge(op) ? (F ? 14ull * H : 0ull) :
                                  op == MeshTopologyOp::SplitConcaveFaces                                                                                                           ? 29ull * H :
                                  op == MeshTopologyOp::SplitNonplanarFaces                                                                                                         ? 14ull * H :
                                  op == MeshTopologyOp::Subdivide                                                                                                                   ? 12ull * H * split_width :
                                                                                                                                                                                      0u);
    job.TableOffset = take(table);
    uint64_t collapse_words = 0;
    if (op == MeshTopologyOp::MergeCollapse && job.CollapseCount) {
        const uint64_t n = job.CollapseCount;
        collapse_words = 3u * n + 16u * ((n + 255u) / 256u) + 16u;
        for (uint64_t level = n;; level = (level + 255u) / 256u) {
            collapse_words += 4u * level;
            if (level <= 256u) break;
        }
    }
    job.CollapseOffset = take(collapse_words);
    job.VertexInwardOffset = take(TopologyDisplaces(op, job.Flags) ? 6ull * bounds.Vertices : 0u);
    return uint32_t(cursor - base);
}
