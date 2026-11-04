#pragma once
// Subdivides the selected edges of a mesh like Blender's edit-mode subdivide.
#include "MeshTopologyContext.metal"
#include "gpu/PartitionDivisions.h"

// A face loop with its cut vertices inserted, each corner carrying its output vertex, its attribute sources, and its edge.
struct SubdivideLoop {
    uint Length;
    device uint *Vertex, *SourceA, *SourceB;
    device float *Weight;
    device uint *EdgeSource; // The source halfedge whose edge the corner's arriving segment belongs to
    device uint *Cut;
    device uint *Selected;
    device uint *Order, *ViaChord, *Visited;
    device packed_uint2 *Chords;
    uint CutEdges;
    uint CutStart; // The expanded index of the first cut on the first selected edge, for pattern rotation
};

inline uint TopoSubdivideCuts(MeshTopologyJob job) { return (job.Flags & TopologyFlagSingleCut) ? 1u : max(1u, uint(job.Param0)); }
inline bool TopoJunctions(MeshTopologyJob job) { return (job.Flags & TopologyFlagListJunctions) != 0u; }
inline uint TopoEdgeCuts(TopoContext ctx, MeshTopologyJob job, uint h) {
    if (job.Flags & TopologyFlagListCounts) return ctx.EdgeParams(job)[ctx.SrcEdge(job, h)];
    return TopoSubdivideCuts(job);
}
inline bool TopoFacePoints(MeshTopologyJob job) { return (job.Flags & TopologyFlagFacePoints) != 0u; }

// Every source corner reserves one original slot and `cuts` cut slots at its halfedge's domain index, so adjacent faces never share scratch.
inline SubdivideLoop TopoSplitLoop(TopoContext ctx, MeshTopologyJob job, uint f) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    const uint width = TopoSubdivideCuts(job) + 1u;
    const uint total = job.SrcHalfedgeCount * width;
    const uint first = ctx.SrcHalfedgeDomain(job).Index(range.x) * width;
    device uint *base = ctx.Scratch() + job.FaceLoopOffset;
    return {0u, base + first, base + total + first, base + 2u * total + first,
        reinterpret_cast<device float *>(base + 3u * total + first),
        base + 4u * total + first, base + 5u * total + first, base + 6u * total + first,
        base + 7u * total + first, base + 8u * total + first, base + 9u * total + first,
        reinterpret_cast<device packed_uint2 *>(base + 10u * total) + first,
        0u, InvalidOffset};
}

inline bool TopoEdgeCut(TopoContext ctx, MeshTopologyJob job, uint h) {
    const uint e = ctx.SrcEdge(job, h);
    if (job.Flags & TopologyFlagListCounts) return ctx.EdgeParams(job)[e] != 0u;
    if (job.Flags & (TopologyFlagScreenCuts | TopologyFlagListJunctions)) return ctx.EdgeParams(job)[e] != InvalidOffset;
    if (job.Flags & TopologyFlagPlaneCuts) {
        const auto corners = ctx.SrcCorners(job);
        const PlaneSideKind a = ctx.PlaneSide(job, ctx.SrcPosition(job, ctx.SrcFrom(job, h)));
        const PlaneSideKind b = ctx.PlaneSide(job, ctx.SrcPosition(job, corners[h]));
        return a != b && a != PlaneSideKind::On && b != PlaneSideKind::On;
    }
    return ctx.SrcSelectedEdge(job, e);
}

inline float TopoCutParam(TopoContext ctx, MeshTopologyJob job, uint rep, uint i, uint cuts) {
    if (job.Flags & TopologyFlagScreenCuts) return as_type<float>(ctx.EdgeParams(job)[ctx.SrcEdge(job, rep)]);
    if (TopoJunctions(job)) {
        const auto corners = ctx.SrcCorners(job);
        const float3 start = ctx.SrcPosition(job, ctx.SrcFrom(job, rep)), d = ctx.SrcPosition(job, corners[rep]) - start;
        return clamp(dot(ctx.SrcPosition(job, ctx.EdgeParams(job)[ctx.SrcEdge(job, rep)]) - start, d) / dot(d, d), 0.f, 1.f);
    }
    if (job.Flags & TopologyFlagPlaneCuts) {
        const auto corners = ctx.SrcCorners(job);
        const float a = ctx.PlaneDistance(job, ctx.SrcPosition(job, ctx.SrcFrom(job, rep)));
        const float b = ctx.PlaneDistance(job, ctx.SrcPosition(job, corners[rep]));
        return clamp(a / (a - b), 0.f, 1.f);
    }
    return float(i + 1u) / float(cuts + 1u);
}

inline void TopoSubdivideExpand(TopoContext ctx, MeshTopologyJob job, uint f, thread SubdivideLoop &loop) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    const auto corners = ctx.SrcCorners(job);
    device const uint *vertex_offsets = ctx.Counts(job, TopoCountVertices);
    loop.Length = 0u;
    loop.CutEdges = 0u;
    loop.CutStart = InvalidOffset;
    for (uint h = range.x; h < range.y; ++h) {
        const uint prev = h == range.x ? range.y - 1u : h - 1u;
        const bool selected = TopoEdgeCut(ctx, job, h);
        if (selected) {
            const uint cuts = TopoEdgeCuts(ctx, job, h);
            ++loop.CutEdges;
            // The cuts run along the edge's representative halfedge, so a corner against it takes them in reverse order.
            const uint rep = ctx.SrcEdgeHalfedge(job, ctx.SrcEdge(job, h));
            const uint first = vertex_offsets[ctx.HalfedgeEntry(job, rep)];
            const bool along = corners[rep] == corners[h];
            if (loop.CutStart == InvalidOffset) loop.CutStart = loop.Length;
            for (uint k = 0u; k < cuts; ++k) {
                const uint i = along ? k : cuts - 1u - k;
                const float along_rep = TopoCutParam(ctx, job, rep, i, cuts);
                const float t = along ? along_rep : 1.f - along_rep;
                const uint slot = loop.Length++;
                loop.Vertex[slot] = TopoJunctions(job) ? vertex_offsets[ctx.EdgeParams(job)[ctx.SrcEdge(job, h)]] : first + i;
                loop.SourceA[slot] = prev;
                loop.SourceB[slot] = h;
                loop.Weight[slot] = t;
                loop.EdgeSource[slot] = h;
                loop.Cut[slot] = true;
                loop.Selected[slot] = true;
            }
        }
        const uint slot = loop.Length++;
        loop.Vertex[slot] = vertex_offsets[corners[h]];
        loop.SourceA[slot] = h;
        loop.SourceB[slot] = h;
        loop.Weight[slot] = 0.f;
        loop.EdgeSource[slot] = h;
        loop.Cut[slot] = false;
        loop.Selected[slot] = selected;
    }
}

// The node grid of a triangle or quad with every edge cut `Cuts` times.
struct SubdivideGrid {
    uint Cuts, InteriorFirst;
    uint4 Corners;
    bool Quad;
    uint RowLength(uint row) const { return Cuts + 2u - (Quad ? 0u : row); }
    uint Node(uint row, uint column) const {
        const uint c = Cuts;
        if (row == 0u) return c + column;
        if (row == c + 1u) return !Quad || column == c + 1u ? 3u * c + 2u : 4u * c + 3u - column;
        if (column == 0u) return c - row;
        if (column == RowLength(row) - 1u) return 2u * c + 1u + row;
        return InvalidOffset;
    }
    uint Interior(uint row, uint column) const {
        const uint preceding = (row - 1u) * Cuts - (Quad ? 0u : (row - 1u) * row / 2u);
        return InteriorFirst + preceding + column - 1u;
    }
    float2 Weights(uint row, uint column) const {
        if (Quad) return float2(float(column) / float(Cuts + 1u), float(row) / float(Cuts + 1u));
        const uint length = RowLength(row);
        return float2(length > 1u ? float(column) / float(length - 1u) : 0.f, float(row) / float(Cuts + 1u));
    }
    uint EdgeSource(uint2 from, uint2 to) const {
        uint edge = InvalidOffset;
        if (to.x == from.x && (to.x == 0u || (Quad && to.x == Cuts + 1u))) edge = to.x == 0u ? Corners.y : Corners.w;
        if (to.y == from.y && (to.y == 0u || (Quad && to.y == Cuts + 1u))) edge = to.y == 0u ? Corners.x : Corners.z;
        if (!Quad && to.y == Cuts + 1u - to.x && from.y == Cuts + 1u - from.x) edge = Corners.z;
        return edge;
    }
};

// Emits one output face of `count` expanded corners in `order`, with chord-arriving corners taking no source edge.
struct SubdivideEmitter {
    TopoContext Ctx;
    MeshTopologyJob Job;
    uint Source;
    bool Emit;
    uint Face, Base; // The next output face and corner
    uint Faces, Corners;
    uint Pair; // The next provenance pair, which the count pass starts at zero for the total

    // Reserves a provenance pair per corner of the source face, weighting corner k by `weight(k)` when emitting, and returns the first.
    template<typename Weight> uint Provenance(bool vertices, Weight weight) {
        const uint2 range = Ctx.SrcFaceRange(Job, Source);
        const uint count = range.y - range.x, first = Pair;
        if (Emit) {
            device packed_uint2 *pairs = Ctx.Provenance(Job);
            const auto corners = Ctx.SrcCorners(Job);
            for (uint k = 0u; k < count; ++k) pairs[first + k] = packed_uint2(vertices ? corners[range.x + k] : range.x + k, as_type<uint>(weight(k)));
        }
        Pair += count;
        return first;
    }

    void Polygon(thread const SubdivideLoop &loop, device const uint *order, uint count, device const uint *chord_arrival, bool selected) {
        if (Emit) {
            for (uint k = 0u; k < count; ++k) {
                const uint i = order ? order[k] : k;
                const bool chord = chord_arrival && chord_arrival[k];
                const bool edge_selected = chord || (loop.Selected[i] && (Job.Flags & TopologyFlagNewLoopsOnly) == 0u);
                Ctx.WriteCorner(Job, Base + k, loop.Vertex[i], loop.SourceA[i], loop.SourceB[i], loop.Weight[i], chord ? InvalidOffset : loop.EdgeSource[i], edge_selected);
            }
            TopoEmitFace(Ctx, Job, Face, Base, count, Source, selected);
        }
        Advance(count);
    }
    template<uint Count>
    void GridPolygon(SubdivideGrid grid, thread const SubdivideLoop &loop, thread const uint2 *cells) {
        if (Emit) {
            for (uint q = 0u; q < Count; ++q) {
                const uint2 cell = cells[q];
                const uint slot = grid.Node(cell.x, cell.y);
                const uint vertex_id = slot != InvalidOffset ? loop.Vertex[slot] : grid.Interior(cell.x, cell.y);
                const uint edge = grid.EdgeSource(cells[(q + Count - 1u) % Count], cell);
                if (slot != InvalidOffset) Ctx.WriteCorner(Job, Base + q, vertex_id, loop.SourceA[slot], loop.SourceB[slot], loop.Weight[slot], edge, true);
                else {
                    const float2 weight = grid.Weights(cell.x, cell.y);
                    Ctx.WriteCorner4(Job, Base + q, vertex_id, grid.Corners, weight.x, weight.y, edge, true);
                }
            }
            TopoEmitFace(Ctx, Job, Face, Base, Count, Source, true);
        }
        Advance(Count);
    }
    void Whole(thread const SubdivideLoop &loop, bool selected) {
        Polygon(loop, nullptr, loop.Length, nullptr, selected);
    }
    void Advance(uint count) {
        ++Face;
        Base += count;
        ++Faces;
        Corners += count;
    }
};

// Splits the loop's polygon along `chord_count` chords between expanded corners and emits every resulting face.
inline void TopoSubdivideByChords(thread SubdivideEmitter &emitter, thread const SubdivideLoop &loop, uint chord_count) {
    const uint n = loop.Length;
    // Loop edge i runs from corner i to i + 1, and chord c runs from x to y in direction 0 and from y to x in direction 1.
    for (uint i = 0u; i < n; ++i) loop.Visited[i] = 0u;
    const auto distance = [&](uint from, uint to) { return (to + n - from) % n; };
    // The outgoing edge from `v` that keeps the face on the left after arriving from `u` has the largest loop distance below `u`'s.
    struct Step {
        uint To;
        bool Chord;
        uint Index; // The chord, or the loop edge's start corner
        uint Direction;
    };
    const auto next_step = [&](uint u, uint v) {
        const uint limit = distance(v, u);
        Step best{(v + 1u) % n, false, v, 0u};
        uint best_distance = distance(v, best.To) < limit ? distance(v, best.To) : 0u;
        for (uint c = 0u; c < chord_count; ++c) {
            const bool forward = loop.Chords[c].x == v;
            if (!forward && loop.Chords[c].y != v) continue;
            const uint other = forward ? loop.Chords[c].y : loop.Chords[c].x;
            const uint d = distance(v, other);
            if (d < limit && d > best_distance) {
                best_distance = d;
                best = {other, true, c, forward ? 0u : 1u};
            }
        }
        return best;
    };
    const auto visited_bit = [](Step step) { return step.Chord ? 2u << step.Direction : 1u; };
    for (uint start = 0u; start < n + 2u * chord_count; ++start) {
        Step first;
        uint u;
        if (start < n) {
            first = {(start + 1u) % n, false, start, 0u};
            u = start;
        } else {
            const uint c = (start - n) / 2u, direction = (start - n) % 2u;
            first = {direction == 0u ? loop.Chords[c].y : loop.Chords[c].x, true, c, direction};
            u = direction == 0u ? loop.Chords[c].x : loop.Chords[c].y;
        }
        if (loop.Visited[first.Index] & visited_bit(first)) continue;
        Step step = first;
        uint count = 0u;
        for (uint guard = 0u; guard < n; ++guard) {
            loop.Visited[step.Index] |= visited_bit(step);
            loop.Order[count] = step.To;
            loop.ViaChord[count] = step.Chord;
            ++count;
            const Step next = next_step(u, step.To);
            if (next.Chord == first.Chord && next.Index == first.Index && next.Direction == first.Direction) break;
            u = step.To;
            step = next;
        }
        emitter.Polygon(loop, loop.Order, count, loop.ViaChord, true);
    }
}

inline uint TopoLoopCornerSlot(thread const SubdivideLoop &loop, uint c) {
    uint slot = 0u, corner = 0u;
    for (; slot < loop.Length; ++slot) {
        if (loop.Cut[slot]) continue;
        if (corner++ == c) return slot;
    }
    return InvalidOffset;
}

// Splits a face whose edges are all cut once into a quad per corner around a face point blending the corners equally.
inline uint TopoSubdivideFacePoint(TopoContext ctx, MeshTopologyJob job, uint f, thread const SubdivideLoop &loop, thread SubdivideEmitter &emitter) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    const uint n = range.y - range.x;
    const uint face_point = ctx.Counts(job, TopoCountVertices)[ctx.FaceEntry(job, f)];
    const float share = 1.f / float(n);
    const uint point_pairs = emitter.Provenance(true, [&](uint) { return share; });
    if (emitter.Emit) {
        ctx.WriteVertexProvenance(job, face_point, point_pairs, n, f);
        ctx.SelectDstVertex(job, face_point);
    }
    const uint mean_pairs = emitter.Provenance(false, [&](uint) { return share; });
    // The expanded loop alternates cut and corner, so the cut before corner c lies on the edge arriving at it.
    for (uint c = 0u; c < n; ++c) {
        const uint corner_slot = 2u * c + 1u, arriving = 2u * c, leaving = (2u * c + 2u) % loop.Length;
        if (emitter.Emit) {
            const uint base = emitter.Base;
            ctx.WriteCorner(job, base, loop.Vertex[corner_slot], loop.SourceA[corner_slot], loop.SourceB[corner_slot], loop.Weight[corner_slot], loop.EdgeSource[corner_slot], true);
            ctx.WriteCorner(job, base + 1u, loop.Vertex[leaving], loop.SourceA[leaving], loop.SourceB[leaving], loop.Weight[leaving], loop.EdgeSource[leaving], true);
            ctx.WriteCornerProvenance(job, base + 2u, face_point, mean_pairs, n, InvalidOffset, true);
            ctx.WriteCorner(job, base + 3u, loop.Vertex[arriving], loop.SourceA[arriving], loop.SourceB[arriving], loop.Weight[arriving], InvalidOffset, true);
            TopoEmitFace(ctx, job, emitter.Face, base, 4u, f, true);
        }
        emitter.Advance(4u);
    }
    return 1u;
}

// A triangle's or quad's partition table in the job's list, holding manifold's triangulation for its sorted side divisions.
struct TopoPartition {
    device const uint *Header; // The four sorted divisions, then the vertex, interior and triangle counts
    device const float *Weights; // Each template vertex's weights over the sorted corners
    device const uint *Triangles;
    uint Vertices() const { return Header[4]; }
    uint Interior() const { return Header[5]; }
    uint Count() const { return Header[6]; }
    uint InteriorOffset() const { return Header[0] + Header[1] + Header[2] + Header[3]; }
    float4 Weight(uint j) const { return float4(Weights[4u * j], Weights[4u * j + 1u], Weights[4u * j + 2u], Weights[4u * j + 3u]); }
};
inline TopoPartition TopoFindPartition(TopoContext ctx, MeshTopologyJob job, uint4 sorted) {
    device const uint *list = ctx.Lists(job);
    device const uint *cursor = list + 1u + 2u * list[0];
    const uint tables = *cursor++;
    for (uint t = 0u; t < tables; ++t) {
        const TopoPartition table{cursor, reinterpret_cast<device const float *>(cursor + 7u), cursor + 7u + 4u * cursor[4]};
        if (cursor[0] == sorted.x && cursor[1] == sorted.y && cursor[2] == sorted.z && cursor[3] == sorted.w) return table;
        cursor += 7u + 4u * cursor[4] + 3u * cursor[6];
    }
    return {nullptr, nullptr, nullptr};
}

// Fills a triangle or quad whose edges are all cut with differing counts by its partition table.
inline uint TopoSubdividePartition(TopoContext ctx, MeshTopologyJob job, uint f, thread const SubdivideLoop &loop, thread SubdivideEmitter &emitter) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    const uint n = range.y - range.x;
    uint4 divisions = uint4(0u);
    for (uint c = 0u; c < n; ++c) divisions[c] = TopoEdgeCuts(ctx, job, range.x + (c + 1u) % n) + 1u;
    const SortedDivisions sorted_divisions = SortDivisions(divisions);
    const uint4 sorted = sorted_divisions.Sorted, order = sorted_divisions.Order;
    const TopoPartition table = TopoFindPartition(ctx, job, sorted);
    // A reflected triangle sort reads its sides backwards and winds its triangles the other way.
    const bool reflected = n == 3u && order.y != (order.x + 1u) % 3u;
    const uint4 corner_of = reflected ? uint4(order.z, order.x, order.y, 3u) : order;
    const uint interior_first = ctx.Counts(job, TopoCountVertices)[ctx.FaceEntry(job, f)];
    const auto output_vertex = [&](uint j) {
        if (j < n) return loop.Vertex[TopoLoopCornerSlot(loop, corner_of[j])];
        if (j >= table.InteriorOffset()) return interior_first + j - table.InteriorOffset();
        uint i = 0u, first = n;
        while (j >= first + sorted[i] - 1u) first += sorted[i++] - 1u;
        const uint edge = order[i], cuts = sorted[i] - 1u, k = j - first;
        const uint start = TopoLoopCornerSlot(loop, (edge + 1u) % n) - cuts;
        return loop.Vertex[start + (reflected ? cuts - 1u - k : k)];
    };
    const auto input_weights = [&](uint j) {
        const float4 w = table.Weight(j);
        float4 weights = float4(0.f);
        for (uint i = 0u; i < n; ++i) weights[corner_of[i]] = w[i];
        return weights;
    };
    const auto side_edge = [&](uint p, uint q) {
        const float4 wp = table.Weight(p), wq = table.Weight(q);
        for (uint i = 0u; i < n; ++i) {
            bool on = true;
            for (uint c = 0u; c < n; ++c) on = on && ((c == i || c == (i + 1u) % n) || (wp[c] == 0.f && wq[c] == 0.f));
            if (on) return range.x + (order[i] + 1u) % n;
        }
        return InvalidOffset;
    };
    for (uint j = table.InteriorOffset(); j < table.Vertices(); ++j) {
        const float4 weights = input_weights(j);
        const uint first = emitter.Provenance(true, [&](uint k) { return weights[k]; });
        if (emitter.Emit) {
            const uint d = output_vertex(j);
            ctx.WriteVertexProvenance(job, d, first, n, f);
            ctx.SelectDstVertex(job, d);
        }
    }
    for (uint t = 0u; t < table.Count(); ++t) {
        uint3 tri = uint3(table.Triangles[3u * t], table.Triangles[3u * t + 1u], table.Triangles[3u * t + 2u]);
        if (reflected) tri = tri.yxz;
        for (uint q = 0u; q < 3u; ++q) {
            const uint j = tri[q], p = tri[(q + 2u) % 3u];
            const float4 weights = input_weights(j);
            const uint first = emitter.Provenance(false, [&](uint k) { return weights[k]; });
            if (emitter.Emit) ctx.WriteCornerProvenance(job, emitter.Base + q, output_vertex(j), first, n, side_edge(p, j), true);
        }
        if (emitter.Emit) TopoEmitFace(ctx, job, emitter.Face, emitter.Base, 3u, f, true);
        emitter.Advance(3u);
    }
    return table.Interior();
}

// Emits Catmull-Clark's vertex mask over the vertex's fan as provenance pairs and returns their count.
template<typename Emit>
inline uint TopoCatmullClarkVertex(TopoContext ctx, MeshTopologyJob job, uint v, Emit emit) {
    const uint2 fan = ctx.SrcFan(job, v);
    const auto src = ctx.Src(job);
    const auto corners = ctx.SrcCorners(job);
    uint boundary = 0u, pairs = 0u;
    uint neighbors[2] = {InvalidOffset, InvalidOffset};
    for (uint i = 0u; i < fan.y; ++i) {
        const uint h = ctx.SrcFanCorner(job, fan.x + i), next = ctx.SrcNext(job, h);
        if (src.Opposite(h) == InvalidOffset) {
            if (boundary < 2u) neighbors[boundary] = ctx.SrcFrom(job, h);
            ++boundary;
        }
        if (src.Opposite(next) == InvalidOffset) {
            if (boundary < 2u) neighbors[boundary] = corners[next];
            ++boundary;
        }
    }
    const uint n = fan.y;
    if (boundary == 0u && n >= 3u) {
        const float share = 1.f / float(n * n);
        for (uint i = 0u; i < n; ++i) {
            const uint h = ctx.SrcFanCorner(job, fan.x + i);
            const uint2 range = ctx.SrcFaceRange(job, ctx.SrcFaceOf(job, h));
            for (uint c = range.x; c < range.y; ++c) emit(pairs++, corners[c], share / float(range.y - range.x));
            emit(pairs++, ctx.SrcFrom(job, h), share);
        }
        emit(pairs++, v, float(n - 2u) / float(n));
        return pairs;
    }
    if (boundary == 2u) {
        emit(pairs++, v, .75f);
        emit(pairs++, neighbors[0], .125f);
        emit(pairs++, neighbors[1], .125f);
        return pairs;
    }
    return 0u;
}
// Emits Catmull-Clark's edge mask for an interior edge's cut over its ends and both faces' corners, and returns the pair count.
template<typename Emit>
inline uint TopoCatmullClarkEdge(TopoContext ctx, MeshTopologyJob job, uint h, Emit emit) {
    const uint opposite = ctx.Src(job).Opposite(h);
    if (opposite == InvalidOffset || ctx.Src(job).HalfedgeFace(opposite) == InvalidOffset) return 0u;
    const auto corners = ctx.SrcCorners(job);
    uint pairs = 0u;
    emit(pairs++, ctx.SrcFrom(job, h), .25f);
    emit(pairs++, corners[h], .25f);
    for (uint side = 0u; side < 2u; ++side) {
        const uint2 range = ctx.SrcFaceRange(job, ctx.SrcFaceOf(job, side ? opposite : h));
        for (uint c = range.x; c < range.y; ++c) emit(pairs++, corners[c], .25f / float(range.y - range.x));
    }
    return pairs;
}
template<typename Count>
inline uint TopoWritePairs(TopoContext ctx, MeshTopologyJob job, uint first, Count count) {
    device packed_uint2 *pairs = ctx.Provenance(job);
    return count([&](uint k, uint source, float weight) { pairs[first + k] = packed_uint2(source, as_type<uint>(weight)); });
}

// Returns the interior vertices the face adds: a grid's or a partition's interior, or one face point.
inline uint TopoSubdivideFace(TopoContext ctx, MeshTopologyJob job, uint f, thread SubdivideEmitter &emitter) {
    SubdivideLoop loop = TopoSplitLoop(ctx, job, f);
    const uint2 range = ctx.SrcFaceRange(job, f);
    const uint n = range.y - range.x;
    TopoSubdivideExpand(ctx, job, f, loop);
    const uint k = loop.CutEdges;
    if (TopoJunctions(job)) {
        emitter.Whole(loop, ctx.SrcSelectedFace(job, f));
        return 0u;
    }
    if (job.Flags & TopologyFlagPlaneCuts) {
        const auto corners = ctx.SrcCorners(job);
        const auto side = [&](uint h) { return ctx.PlaneSide(job, ctx.SrcPosition(job, corners[h])); };
        uint crossings[2], count = 0u;
        for (uint i = 0u; i < loop.Length; ++i) {
            if (!loop.Cut[i]) {
                const uint h = loop.SourceA[i];
                if (side(h) != PlaneSideKind::On) continue;
                const PlaneSideKind before = side(ctx.SrcPrev(job, h)), after = side(ctx.SrcNext(job, h));
                if (before == after || before == PlaneSideKind::On || after == PlaneSideKind::On) continue;
            }
            if (count < 2u) crossings[count] = i;
            ++count;
        }
        if (count == 2u) {
            loop.Chords[0] = packed_uint2(crossings[0], crossings[1]);
            TopoSubdivideByChords(emitter, loop, 1u);
            return 0u;
        }
    }
    if (k == n && TopoFacePoints(job)) return TopoSubdivideFacePoint(ctx, job, f, loop, emitter);
    uint cuts = 0u;
    bool uniform = true;
    for (uint h = range.x; h < range.y; ++h) {
        if (!TopoEdgeCut(ctx, job, h)) continue;
        const uint count = TopoEdgeCuts(ctx, job, h);
        uniform = uniform && (cuts == 0u || count == cuts);
        cuts = count;
    }
    if (!uniform && k == n && (n == 3u || n == 4u)) return TopoSubdividePartition(ctx, job, f, loop, emitter);
    if (k < 2u || !uniform || (k > 2u && n != 4u && n != 3u)) {
        emitter.Whole(loop, ctx.SrcSelectedFace(job, f));
        return 0u;
    }
    if ((n == 4u || n == 3u) && k == n) {
        const bool quad = n == 4u;
        const uint c = cuts;
        const SubdivideGrid grid{c, ctx.Counts(job, TopoCountVertices)[ctx.FaceEntry(job, f)],
            uint4(range.x, range.x + 1u, range.x + 2u, range.x + (quad ? 3u : 2u)), quad};
        const auto corners = ctx.SrcCorners(job);
        const uint4 sources = uint4(corners[grid.Corners.x], corners[grid.Corners.y], corners[grid.Corners.z], corners[grid.Corners.w]);
        for (uint row = 1u; row < (quad ? c + 1u : c); ++row) {
            for (uint column = 1u; column + 1u < grid.RowLength(row); ++column) {
                const uint d = grid.Interior(row, column);
                const float2 weight = grid.Weights(row, column);
                if (job.Flags & TopologyFlagBezier) {
                    const float4 weights = TopoBilinearWeights(weight.x, weight.y);
                    const uint first = emitter.Provenance(true, [&](uint k) { return quad ? weights[k] : (k == 2u ? weights.z + weights.w : weights[k]); });
                    if (emitter.Emit) {
                        ctx.WriteVertexProvenance(job, d, first, n, f);
                        ctx.SelectDstVertex(job, d);
                    }
                } else if (emitter.Emit) {
                    ctx.WriteVertexMap4(job, d, sources, weight.x, weight.y);
                    ctx.SelectDstVertex(job, d);
                }
            }
        }
        for (uint row = 0u; row <= c; ++row) {
            const uint length = grid.RowLength(row);
            for (uint column = 0u; column + 1u < length; ++column) {
                const uint2 a{row, column}, b{row, column + 1u}, d{row + 1u, column};
                if (quad) {
                    const uint2 cells[4] = {a, b, uint2(row + 1u, column + 1u), d};
                    emitter.GridPolygon<4u>(grid, loop, cells);
                } else {
                    const uint2 cells[3] = {a, b, d};
                    emitter.GridPolygon<3u>(grid, loop, cells);
                    if (column + 2u < length) {
                        const uint2 next[3] = {b, uint2(row + 1u, column + 1u), d};
                        emitter.GridPolygon<3u>(grid, loop, next);
                    }
                }
            }
        }
        return quad ? c * c : c > 1u ? (c - 1u) * c / 2u : 0u;
    }
    uint chord_count = 0u;
    if (k == 2u) {
        const uint a = loop.CutStart;
        uint b = (a + cuts) % loop.Length;
        for (uint step = 0u; step < loop.Length && !(loop.Cut[b] && !loop.Cut[(b + loop.Length - 1u) % loop.Length]); ++step) b = (b + 1u) % loop.Length;
        const uint b_last = (b + cuts - 1u) % loop.Length;
        for (uint j = 0u; j < cuts; ++j) {
            loop.Chords[chord_count++] = packed_uint2((a + j) % loop.Length, (b_last + loop.Length - j) % loop.Length);
        }
    } else {
        // Blender's three-edge quad pattern indexes the expanded loop from the first cut of the run's first selected edge.
        uint rotation = loop.CutStart;
        for (uint corner = 0u; corner < loop.Length; ++corner) {
            const uint i = (loop.CutStart + corner) % loop.Length;
            const uint before = (i + loop.Length - 1u) % loop.Length;
            if (loop.Cut[i] && !loop.Cut[before] && !loop.Selected[before]) {
                rotation = i;
                break;
            }
        }
        const auto at = [&](uint index) { return (rotation + index) % loop.Length; };
        const uint c = cuts;
        uint add = 0u;
        for (uint i = 0u; i < c; ++i) {
            if (i == c / 2u) {
                if (c % 2u != 0u) loop.Chords[chord_count++] = packed_uint2(at(c - i - 1u + add), at(i + c + 1u));
                add = c * 2u + 2u;
            }
            loop.Chords[chord_count++] = packed_uint2(at(c - i - 1u + add), at(i + c + 1u));
        }
        for (uint i = 0u; i < c / 2u + 1u; ++i) {
            loop.Chords[chord_count++] = packed_uint2(at(i), at((c - i) + c * 2u + 1u));
        }
    }
    TopoSubdivideByChords(emitter, loop, chord_count);
    return 0u;
}

// Connect and edge rotation split a source or dissolved loop between consecutive selected corners by walking its boundary.
inline uint TopoConnectNext(TopoContext ctx, MeshTopologyJob job, uint h, bool own_loop) {
    return own_loop ? ctx.SrcNext(job, h) : TopoNextBoundary(ctx, job, h);
}
inline bool TopoConnectKept(TopoContext ctx, MeshTopologyJob job, uint h) {
    return !TopoVertexRemoved(ctx, job, ctx.SrcCorners(job)[h]);
}
struct TopoConnectMeasure {
    uint Start, FirstSelected, Length, Selected, Ears, EarCorners, FirstGap;
};
inline TopoConnectMeasure TopoMeasureConnect(TopoContext ctx, MeshTopologyJob job, uint f, bool own_loop) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    const uint start = own_loop ? range.x : ctx.RegionStart(job)[f];
    const uint limit = own_loop ? range.y - range.x : ctx.RegionBoundary(job)[f];
    TopoConnectMeasure result{start, InvalidOffset, 0u, 0u, 0u, 0u, 0u};
    uint previous_selected = 0u, first_selected = 0u, h = start, steps = 0u;
    if (start == InvalidOffset || limit == 0u) return result;
    do {
        if (++steps > limit) return {start, InvalidOffset, 0u, 0u, 0u, 0u, 0u};
        if (TopoConnectKept(ctx, job, h)) {
            if (ctx.SrcSelectedVertex(job, ctx.SrcCorners(job)[h])) {
                if (result.Selected == 0u) {
                    result.FirstSelected = h;
                    first_selected = result.Length;
                } else {
                    const uint gap = result.Length - previous_selected;
                    if (gap >= 2u) { ++result.Ears; result.EarCorners += gap + 1u; }
                }
                previous_selected = result.Length;
                ++result.Selected;
            }
            ++result.Length;
        }
        h = TopoConnectNext(ctx, job, h, own_loop);
    } while (h != start);
    if (steps != limit) return {start, InvalidOffset, 0u, 0u, 0u, 0u, 0u};
    if (result.Selected > 1u) {
        result.FirstGap = first_selected + result.Length - previous_selected;
        if (result.FirstGap >= 2u) { ++result.Ears; result.EarCorners += result.FirstGap + 1u; }
    }
    return result;
}

inline uint2 TopoConnectOutputs(TopoConnectMeasure m) {
    if (m.Length < 3u) return uint2(0u);
    if (m.Selected < 3u) return m.Selected == 2u && m.Ears == 2u ? uint2(2u, m.EarCorners) : uint2(1u, m.Length);
    return uint2(m.Ears + 1u, m.EarCorners + m.Selected);
}

inline void TopoConnectWriteCorner(TopoContext ctx, MeshTopologyJob job, uint dst, uint h, bool chord) {
    const uint v = ctx.SrcCorners(job)[h];
    const uint output = ctx.Counts(job, TopoCountVertices)[v];
    ctx.WriteCorner(job, dst, output, h, h, 0.f, chord ? InvalidOffset : h,
        chord || (ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, h)) &&
            (job.Op != MeshTopologyOp::ConnectVertices || (job.Flags & TopologyFlagNewLoopsOnly) == 0u)));
}

inline uint TopoConnectWriteEar(TopoContext ctx, MeshTopologyJob job, uint f, bool own_loop,
                               uint first, uint gap, uint face, uint base) {
    uint h = first;
    uint written = 0u;
    while (written <= gap) {
        if (TopoConnectKept(ctx, job, h)) {
            TopoConnectWriteCorner(ctx, job, base + written, h, written == 0u);
            ++written;
        }
        h = TopoConnectNext(ctx, job, h, own_loop);
    }
    TopoEmitFace(ctx, job, face, base, written, f, true);
    return written;
}

inline void TopoEmitConnect(TopoContext ctx, MeshTopologyJob job, uint f, bool own_loop,
                             TopoConnectMeasure m, uint face, uint base) {
    const uint2 output = TopoConnectOutputs(m);
    if (output.x == 0u) return;
    const uint start = m.Selected ? m.FirstSelected : m.Start;
    if (output.x == 1u && (m.Selected < 3u || m.Ears == 0u)) {
        const uint whole_start = job.Op == MeshTopologyOp::ConnectVertices ? m.Start : start;
        uint h = whole_start, written = 0u;
        do {
            if (TopoConnectKept(ctx, job, h)) TopoConnectWriteCorner(ctx, job, base + written++, h, false);
            h = TopoConnectNext(ctx, job, h, own_loop);
        } while (h != whole_start);
        const uint2 source_range = ctx.SrcFaceRange(job, f);
        const bool selected = own_loop ? ctx.SrcSelectedFace(job, f) :
            ctx.SrcSelectedFace(job, f) || ctx.RegionBoundary(job)[f] != source_range.y - source_range.x;
        TopoEmitFace(ctx, job, face, base, written, f, selected);
        return;
    }
    uint h = start, previous = start, gap = 0u;
    do {
        if (TopoConnectKept(ctx, job, h)) {
            if (ctx.SrcSelectedVertex(job, ctx.SrcCorners(job)[h]) && h != start) {
                if (gap >= 2u) {
                    base += TopoConnectWriteEar(ctx, job, f, own_loop, previous, gap, face++, base);
                }
                previous = h;
                gap = 0u;
            }
            ++gap;
        }
        h = TopoConnectNext(ctx, job, h, own_loop);
    } while (h != start);
    if (gap >= 2u) base += TopoConnectWriteEar(ctx, job, f, own_loop, previous, gap, face++, base);
    if (m.Selected < 3u) return;
    h = start;
    uint written = 0u;
    gap = m.FirstGap;
    do {
        if (TopoConnectKept(ctx, job, h)) {
            if (ctx.SrcSelectedVertex(job, ctx.SrcCorners(job)[h])) {
                TopoConnectWriteCorner(ctx, job, base + written++, h, gap >= 2u);
                gap = 0u;
            }
            ++gap;
        }
        h = TopoConnectNext(ctx, job, h, own_loop);
    } while (h != start);
    TopoEmitFace(ctx, job, face, base, written, f, true);
}
