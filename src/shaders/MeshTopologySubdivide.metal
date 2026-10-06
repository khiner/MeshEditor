#ifndef MESHTOPOLOGYSUBDIVIDE_MSL
#define MESHTOPOLOGYSUBDIVIDE_MSL

// Subdivides the selected edges of a mesh like Blender's edit-mode subdivide.
// Every selected edge takes `cuts` vertices.
// A face with two cut edges connects the cuts with chords, and a quad with three cut edges follows Blender's chord pattern.
// A quad or triangle with every edge cut fills with a grid, and any other face keeps one loop with the cuts inserted.
#include "MeshTopologyContext.metal"

// A face loop with its cut vertices inserted, each corner carrying its output vertex, its attribute sources, and its edge.
struct SubdivideLoop {
    uint Length;
    device uint *Vertex, *SourceA, *SourceB;
    device float *Weight;
    device uint *EdgeSource; // The source halfedge whose edge the corner's arriving segment belongs to
    device uint *Cut; // A cut vertex rather than an original corner
    device uint *Selected; // The arriving segment is a piece of a selected edge
    device uint *Order, *ViaChord, *Visited;
    device packed_uint2 *Chords;
    uint CutEdges; // Selected edges around the face
    uint CutStart; // The expanded index of the first cut on the first selected edge, for pattern rotation
};

inline uint TopoSubdivideCuts(MeshTopologyJob job) { return (job.Flags & (TopologyFlagPlaneCuts | TopologyFlagListCuts | TopologyFlagScreenCuts)) ? 1u : max(1u, uint(job.Param0)); }

// Each face owns the portion of the affected halfedge work that contains its
// source loop. Every source corner reserves one original slot and `cuts` cut
// slots, so adjacent faces never share scratch even when handles are sparse.
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
    if (job.Flags & (TopologyFlagListCuts | TopologyFlagScreenCuts)) return ctx.EdgeParams(job)[e] != InvalidOffset;
    if (job.Flags & TopologyFlagPlaneCuts) {
        const auto corners = ctx.SrcCorners(job);
        const float a = ctx.PlaneDistance(job, ctx.SrcPosition(job, corners[ctx.SrcPrev(job, h)]));
        const float b = ctx.PlaneDistance(job, ctx.SrcPosition(job, corners[h]));
        return (a < 0.f) != (b < 0.f) && a != b;
    }
    return ctx.SrcSelectedEdge(job, e);
}

// The parameter of cut `i` along the representative halfedge `rep`, from its start.
inline float TopoCutParam(TopoContext ctx, MeshTopologyJob job, uint rep, uint i, uint cuts) {
    if (job.Flags & (TopologyFlagListCuts | TopologyFlagScreenCuts)) return as_type<float>(ctx.EdgeParams(job)[ctx.SrcEdge(job, rep)]);
    if (job.Flags & TopologyFlagPlaneCuts) {
        const auto corners = ctx.SrcCorners(job);
        const float a = ctx.PlaneDistance(job, ctx.SrcPosition(job, corners[ctx.SrcPrev(job, rep)]));
        const float b = ctx.PlaneDistance(job, ctx.SrcPosition(job, corners[rep]));
        return clamp(a / (a - b), 0.f, 1.f);
    }
    return float(i + 1u) / float(cuts + 1u);
}

// Builds the face's expanded loop in its exactly sized source-work slice.
inline void TopoSubdivideExpand(TopoContext ctx, MeshTopologyJob job, uint f, thread SubdivideLoop &loop) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    const uint cuts = TopoSubdivideCuts(job);
    const auto corners = ctx.SrcCorners(job);
    device const uint *vertex_offsets = ctx.Counts(job, TopoCountVertices);
    loop.Length = 0u;
    loop.CutEdges = 0u;
    loop.CutStart = InvalidOffset;
    for (uint h = range.x; h < range.y; ++h) {
        const uint prev = h == range.x ? range.y - 1u : h - 1u;
        const bool selected = TopoEdgeCut(ctx, job, h);
        if (selected) {
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
                loop.Vertex[slot] = first + i;
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

// The triangle and quad grids share generated vertex and corner provenance.
// Rows shorten only for triangles; boundary nodes retain the expanded loop's attributes.
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
    uint Source; // The source face
    bool Emit;
    uint Face, Base; // The next output face and corner
    uint Faces, Corners; // Totals so far

    void Polygon(thread const SubdivideLoop &loop, device const uint *order, uint count, device const uint *chord_arrival, bool selected) {
        if (Emit) {
            for (uint k = 0u; k < count; ++k) {
                const uint i = order ? order[k] : k;
                const bool chord = chord_arrival && chord_arrival[k];
                // A loop cut selects only the new loop, and a subdivide also keeps the cut edges' segments selected.
                const bool edge_selected = chord || (loop.Selected[i] && (Job.Flags & TopologyFlagLoopCutSelect) == 0u);
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
    // Emits the loop as one face in its own order.
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
// Corners lie in loop order, so around any corner the other endpoints sort by loop distance, which orders the faces around it.
inline void TopoSubdivideByChords(thread SubdivideEmitter &emitter, thread const SubdivideLoop &loop, uint chord_count) {
    const uint n = loop.Length;
    // Directed edges: loop edge i runs i -> i + 1, and chord c runs x -> y as direction 0 and y -> x as direction 1.
    // Loop edges and each chord direction have distinct bits at their index.
    // Every generated chord list fits within the expanded loop's length.
    for (uint i = 0u; i < n; ++i) loop.Visited[i] = 0u;
    const auto distance = [&](uint from, uint to) { return (to + n - from) % n; };
    // The outgoing edge from `v` that keeps the face on the left after arriving from `u`: the largest loop distance below `u`'s.
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
    // Every loop edge and every chord direction starts one face walk, in a fixed order, unless a walk already covered it.
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

// Emits the faces of a subdivided face: its pattern's split, or its expanded loop as one face.
// Returns the interior vertices the face adds, which only a grid fill uses.
inline uint TopoSubdivideFace(TopoContext ctx, MeshTopologyJob job, uint f, thread SubdivideEmitter &emitter) {
    SubdivideLoop loop = TopoSplitLoop(ctx, job, f);
    const uint2 range = ctx.SrcFaceRange(job, f);
    const uint n = range.y - range.x;
    const uint cuts = TopoSubdivideCuts(job);
    TopoSubdivideExpand(ctx, job, f, loop);
    const uint k = loop.CutEdges;
    if (k < 2u || (k > 2u && n != 4u && n != 3u)) {
        emitter.Whole(loop, ctx.SrcSelectedFace(job, f));
        return 0u;
    }
    if ((n == 4u || n == 3u) && k == n) {
        const bool quad = n == 4u;
        const uint c = cuts;
        const SubdivideGrid grid{c, ctx.Counts(job, TopoCountVertices)[ctx.FaceEntry(job, f)],
            uint4(range.x, range.x + 1u, range.x + 2u, range.x + (quad ? 3u : 2u)), quad};
        if (emitter.Emit) {
            const auto corners = ctx.SrcCorners(job);
            const uint4 sources = uint4(corners[grid.Corners.x], corners[grid.Corners.y], corners[grid.Corners.z], corners[grid.Corners.w]);
            for (uint row = 1u; row < (quad ? c + 1u : c); ++row) {
                for (uint column = 1u; column + 1u < grid.RowLength(row); ++column) {
                    const uint d = grid.Interior(row, column);
                    const float2 weight = grid.Weights(row, column);
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
        // The first run of cuts pairs with the second run in reverse, from the run ends nearest each other.
        const uint a = loop.CutStart;
        uint b = (a + cuts) % loop.Length;
        for (uint step = 0u; step < loop.Length && !(loop.Cut[b] && !loop.Cut[(b + loop.Length - 1u) % loop.Length]); ++step) b = (b + 1u) % loop.Length;
        const uint b_last = (b + cuts - 1u) % loop.Length;
        for (uint j = 0u; j < cuts; ++j) {
            loop.Chords[chord_count++] = packed_uint2((a + j) % loop.Length, (b_last + loop.Length - j) % loop.Length);
        }
    } else {
        // Blender's three-edge quad: the expanded loop rotated so index 0 is the first cut of the first selected edge of the run.
        uint rotation = loop.CutStart;
        // The run of three selected edges begins at the edge whose predecessor is unselected.
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

// Connect and edge rotation split consecutive selected corners around a source or dissolved loop.
// Walk the boundary directly: no expanded loop or chord graph is needed, regardless of region size.
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
            (job.Op != MeshTopologyOp::ConnectVertices || (job.Flags & TopologyFlagLoopCutSelect) == 0u)));
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

#endif
