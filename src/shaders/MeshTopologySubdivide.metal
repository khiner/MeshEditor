#ifndef MESHTOPOLOGYSUBDIVIDE_MSL
#define MESHTOPOLOGYSUBDIVIDE_MSL

// Subdivides the selected edges of a mesh like Blender's edit-mode subdivide.
// Every selected edge takes `cuts` vertices.
// A face with two cut edges connects the cuts with chords, and a quad with three cut edges follows Blender's chord pattern.
// A quad or triangle with every edge cut fills with a grid, and any other face keeps one loop with the cuts inserted.
#include "MeshTopologyContext.metal"

constant uint SubdivideMaxCorners = 64u;
constant uint SubdivideMaxChords = 32u;

// A face loop with its cut vertices inserted, each corner carrying its output vertex, its attribute sources, and its edge.
struct SubdivideLoop {
    uint Length;
    uint Vertex[SubdivideMaxCorners];
    uint SourceA[SubdivideMaxCorners], SourceB[SubdivideMaxCorners];
    float Weight[SubdivideMaxCorners];
    uint EdgeSource[SubdivideMaxCorners]; // The source halfedge whose edge the corner's arriving segment belongs to
    bool Cut[SubdivideMaxCorners]; // A cut vertex rather than an original corner
    bool Selected[SubdivideMaxCorners]; // The arriving segment is a piece of a selected edge
    uint CutEdges; // Selected edges around the face
    uint CutStart; // The expanded index of the first cut on the first selected edge, for pattern rotation
};

inline uint TopoSubdivideCuts(MeshTopologyJob job) { return (job.Flags & (TopologyFlagPlaneCuts | TopologyFlagListCuts | TopologyFlagScreenCuts)) ? 1u : max(1u, uint(job.Param0)); }

inline bool TopoEdgeCut(TopoContext ctx, MeshTopologyJob job, uint h) {
    const uint e = ctx.SrcEdge(job, h);
    if (job.Flags & (TopologyFlagListCuts | TopologyFlagScreenCuts)) return ctx.EdgeParams(job)[e] != InvalidOffset;
    if (job.Flags & TopologyFlagPlaneCuts) {
        device const uint *corners = ctx.SrcCorners(job);
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
        device const uint *corners = ctx.SrcCorners(job);
        const float a = ctx.PlaneDistance(job, ctx.SrcPosition(job, corners[ctx.SrcPrev(job, rep)]));
        const float b = ctx.PlaneDistance(job, ctx.SrcPosition(job, corners[rep]));
        return clamp(a / (a - b), 0.f, 1.f);
    }
    return float(i + 1u) / float(cuts + 1u);
}

// Builds the face's expanded loop, or returns false when it would not fit.
inline bool TopoSubdivideExpand(TopoContext ctx, MeshTopologyJob job, uint f, thread SubdivideLoop &loop) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    const uint cuts = TopoSubdivideCuts(job);
    device const uint *corners = ctx.SrcCorners(job);
    device const uint *vertex_offsets = ctx.Counts(job, TopoCountVertices);
    loop.Length = 0u;
    loop.CutEdges = 0u;
    loop.CutStart = InvalidOffset;
    for (uint h = range.x; h < range.y; ++h) {
        const uint prev = h == range.x ? range.y - 1u : h - 1u;
        const bool selected = TopoEdgeCut(ctx, job, h);
        if (selected) {
            if (loop.Length + cuts + 1u > SubdivideMaxCorners) return false;
            ++loop.CutEdges;
            // The cuts run along the edge's representative halfedge, so a corner against it takes them in reverse order.
            const uint rep = ctx.SrcEdgeHalfedge(job, ctx.SrcEdge(job, h));
            const uint first = vertex_offsets[ctx.HalfedgeEntry(job, rep)];
            const bool along = rep == h;
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
        } else if (loop.Length + 1u > SubdivideMaxCorners) {
            return false;
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
    return true;
}

// Emits one output face of `count` expanded corners in `order`, with chord-arriving corners taking no source edge.
struct SubdivideEmitter {
    TopoContext Ctx;
    MeshTopologyJob Job;
    uint Source; // The source face
    bool Emit;
    uint Face, Base; // The next output face and corner
    uint Faces, Corners; // Totals so far

    void Polygon(thread const SubdivideLoop &loop, thread const uint *order, uint count, thread const bool *chord_arrival, bool selected) {
        if (Emit) {
            for (uint k = 0u; k < count; ++k) {
                const uint i = order[k];
                // A loop cut selects only the new loop, and a subdivide also keeps the cut edges' segments selected.
                const bool edge_selected = chord_arrival[k] || (loop.Selected[i] && (Job.Flags & TopologyFlagLoopCutSelect) == 0u);
                Ctx.WriteCorner(Job, Base + k, loop.Vertex[i], loop.SourceA[i], loop.SourceB[i], loop.Weight[i], chord_arrival[k] ? InvalidOffset : loop.EdgeSource[i], edge_selected);
            }
            TopoEmitFace(Ctx, Job, Face, Base, Source, selected);
        }
        Advance(count);
    }
    // Emits the loop as one face in its own order.
    void Whole(thread const SubdivideLoop &loop, bool selected) {
        uint order[SubdivideMaxCorners];
        bool via_chord[SubdivideMaxCorners];
        for (uint i = 0u; i < loop.Length; ++i) {
            order[i] = i;
            via_chord[i] = false;
        }
        Polygon(loop, order, loop.Length, via_chord, selected);
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
inline void TopoSubdivideByChords(thread SubdivideEmitter &emitter, thread const SubdivideLoop &loop, thread const uint2 *chords, uint chord_count) {
    const uint n = loop.Length;
    // Directed edges: loop edge i runs i -> i + 1, and chord c runs x -> y as direction 0 and y -> x as direction 1.
    uint visited_loop[2] = {0u, 0u};
    uint visited_chord[2] = {0u, 0u};
    uint order[SubdivideMaxCorners];
    bool via_chord[SubdivideMaxCorners];
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
            const bool forward = chords[c].x == v;
            if (!forward && chords[c].y != v) continue;
            const uint other = forward ? chords[c].y : chords[c].x;
            const uint d = distance(v, other);
            if (d < limit && d > best_distance) {
                best_distance = d;
                best = {other, true, c, forward ? 0u : 1u};
            }
        }
        return best;
    };
    const auto visited = [&](Step step) {
        if (step.Chord) return (visited_chord[step.Direction] >> step.Index) & 1u;
        return (visited_loop[step.Index >> 5u] >> (step.Index & 31u)) & 1u;
    };
    const auto visit = [&](Step step) {
        if (step.Chord) visited_chord[step.Direction] |= 1u << step.Index;
        else visited_loop[step.Index >> 5u] |= 1u << (step.Index & 31u);
    };
    // Every loop edge and every chord direction starts one face walk, in a fixed order, unless a walk already covered it.
    for (uint start = 0u; start < n + 2u * chord_count; ++start) {
        Step first;
        uint u;
        if (start < n) {
            first = {(start + 1u) % n, false, start, 0u};
            u = start;
        } else {
            const uint c = (start - n) / 2u, direction = (start - n) % 2u;
            first = {direction == 0u ? chords[c].y : chords[c].x, true, c, direction};
            u = direction == 0u ? chords[c].x : chords[c].y;
        }
        if (visited(first)) continue;
        Step step = first;
        uint count = 0u;
        for (uint guard = 0u; guard < SubdivideMaxCorners; ++guard) {
            visit(step);
            order[count] = step.To;
            via_chord[count] = step.Chord;
            ++count;
            const Step next = next_step(u, step.To);
            if (next.Chord == first.Chord && next.Index == first.Index && next.Direction == first.Direction) break;
            u = step.To;
            step = next;
        }
        emitter.Polygon(loop, order, count, via_chord, true);
    }
}

// Fills the loop with the face's own corners on their vertices' outputs, with each segment's source edge selection kept when `edges_selected`.
inline void TopoPlainLoop(TopoContext ctx, MeshTopologyJob job, uint f, thread SubdivideLoop &loop, bool edges_selected) {
    const uint2 range = ctx.SrcFaceRange(job, f);
    device const uint *corners = ctx.SrcCorners(job);
    device const uint *vertex_offsets = ctx.Counts(job, TopoCountVertices);
    loop.Length = min(range.y - range.x, SubdivideMaxCorners);
    for (uint k = 0u; k < loop.Length; ++k) {
        const uint h = range.x + k;
        loop.Vertex[k] = vertex_offsets[corners[h]];
        loop.SourceA[k] = loop.SourceB[k] = loop.EdgeSource[k] = h;
        loop.Weight[k] = 0.f;
        loop.Cut[k] = false;
        loop.Selected[k] = edges_selected && ctx.SrcSelectedEdge(job, ctx.SrcEdge(job, h));
    }
}

// Emits the faces of a subdivided face: its pattern's split, or its expanded loop as one face.
// Returns the interior vertices the face adds, which only a grid fill uses.
inline uint TopoSubdivideFace(TopoContext ctx, MeshTopologyJob job, uint f, thread SubdivideEmitter &emitter) {
    SubdivideLoop loop;
    const uint2 range = ctx.SrcFaceRange(job, f);
    const uint n = range.y - range.x;
    const uint cuts = TopoSubdivideCuts(job);
    if (!TopoSubdivideExpand(ctx, job, f, loop)) {
        // A loop too long to expand is emitted unchanged.
        TopoPlainLoop(ctx, job, f, loop, false);
        emitter.Whole(loop, ctx.SrcSelectedFace(job, f));
        return 0u;
    }
    const uint k = loop.CutEdges;
    if (k < 2u || (k > 2u && n != 4u && n != 3u)) {
        emitter.Whole(loop, ctx.SrcSelectedFace(job, f));
        return 0u;
    }
    if (n == 4u && k == 4u) {
        // A grid of cuts + 2 rows over the expanded loop, which starts with the cuts of the edge arriving at corner 0.
        const uint c = cuts;
        const uint interior_first = ctx.Counts(job, TopoCountVertices)[ctx.FaceEntry(job, f)];
        const uint h0 = range.x, h1 = range.x + 1u, h2 = range.x + 2u, h3 = range.x + 3u;
        const auto node = [&](uint i, uint j) -> uint {
            if (i == 0u) return c + j;
            if (i == c + 1u) return j == c + 1u ? 3u * c + 2u : 4u * c + 3u - j;
            if (j == 0u) return c - i;
            if (j == c + 1u) return 2u * c + 1u + i;
            return InvalidOffset;
        };
        const auto vertex_of = [&](uint i, uint j) -> uint {
            const uint slot = node(i, j);
            return slot != InvalidOffset ? loop.Vertex[slot] : interior_first + (i - 1u) * c + (j - 1u);
        };
        if (emitter.Emit) {
            for (uint i = 1u; i <= c; ++i) {
                for (uint j = 1u; j <= c; ++j) {
                    const uint d = interior_first + (i - 1u) * c + (j - 1u);
                    ctx.WriteVertexMap4(job, d, uint4(ctx.SrcCorners(job)[h0], ctx.SrcCorners(job)[h1], ctx.SrcCorners(job)[h2], ctx.SrcCorners(job)[h3]), float(j) / float(c + 1u), float(i) / float(c + 1u));
                    ctx.Select(job.DstVertexBitsOffset, d);
                }
            }
        }
        for (uint i = 0u; i <= c; ++i) {
            for (uint j = 0u; j <= c; ++j) {
                const uint2 cells[4] = {uint2(i, j), uint2(i, j + 1u), uint2(i + 1u, j + 1u), uint2(i + 1u, j)};
                if (emitter.Emit) {
                    for (uint q = 0u; q < 4u; ++q) {
                        const uint2 cell = cells[q], from = cells[(q + 3u) % 4u];
                        const uint hd = emitter.Base + q, v_out = vertex_of(cell.x, cell.y);
                        const uint slot = node(cell.x, cell.y);
                        // A segment along one boundary row or column belongs to that source edge.
                        uint edge_source = InvalidOffset;
                        if (cell.x == from.x && (cell.x == 0u || cell.x == c + 1u)) edge_source = cell.x == 0u ? h1 : h3;
                        if (cell.y == from.y && (cell.y == 0u || cell.y == c + 1u)) edge_source = cell.y == 0u ? h0 : h2;
                        if (slot != InvalidOffset) ctx.WriteCorner(job, hd, v_out, loop.SourceA[slot], loop.SourceB[slot], loop.Weight[slot], edge_source, true);
                        else ctx.WriteCorner4(job, hd, v_out, uint4(h0, h1, h2, h3), float(cell.y) / float(c + 1u), float(cell.x) / float(c + 1u), edge_source, true);
                    }
                    TopoEmitFace(ctx, job, emitter.Face, emitter.Base, f, true);
                }
                emitter.Advance(4u);
            }
        }
        return c * c;
    }
    if (n == 3u && k == 3u) {
        // Rows from the edge arriving at corner 1 up to corner 2: row r holds cuts + 2 - r nodes.
        const uint c = cuts;
        const uint interior_first = ctx.Counts(job, TopoCountVertices)[ctx.FaceEntry(job, f)];
        const uint h0 = range.x, h1 = range.x + 1u, h2 = range.x + 2u;
        // Expanded loop: cuts of h0 (v2 -> v0), v0 at c, cuts of h1 (v0 -> v1), v1 at 2c + 1, cuts of h2 (v1 -> v2), v2 at 3c + 2.
        const auto node = [&](uint r, uint j) -> uint {
            const uint len = c + 2u - r;
            if (r == 0u) return c + j;
            if (r == c + 1u) return 3u * c + 2u;
            if (j == 0u) return c - r;
            if (j == len - 1u) return 2u * c + 1u + r;
            return InvalidOffset;
        };
        const auto interior_index = [&](uint r, uint j) -> uint {
            // Interior nodes of row r (1 <= r <= c - 1) number c - r, packed row by row.
            uint index = 0u;
            for (uint q = 1u; q < r; ++q) index += c - q;
            return interior_first + index + j - 1u;
        };
        const auto vertex_of = [&](uint r, uint j) -> uint {
            const uint slot = node(r, j);
            return slot != InvalidOffset ? loop.Vertex[slot] : interior_index(r, j);
        };
        const auto weights = [&](uint r, uint j) -> float2 {
            const uint len = c + 2u - r;
            return float2(len > 1u ? float(j) / float(len - 1u) : 0.f, float(r) / float(c + 1u));
        };
        const uint4 sources = uint4(ctx.SrcCorners(job)[h0], ctx.SrcCorners(job)[h1], ctx.SrcCorners(job)[h2], ctx.SrcCorners(job)[h2]);
        if (emitter.Emit) {
            for (uint r = 1u; r < c; ++r) {
                for (uint j = 1u; j + 1u < c + 2u - r; ++j) {
                    const uint d = interior_index(r, j);
                    const float2 w = weights(r, j);
                    ctx.WriteVertexMap4(job, d, sources, w.x, w.y);
                    ctx.Select(job.DstVertexBitsOffset, d);
                }
            }
        }
        const auto triangle = [&](uint2 a, uint2 b, uint2 cc) {
            if (emitter.Emit) {
                const uint2 cells[3] = {a, b, cc};
                for (uint q = 0u; q < 3u; ++q) {
                    const uint2 cell = cells[q], from = cells[(q + 2u) % 3u];
                    const uint hd = emitter.Base + q, v_out = vertex_of(cell.x, cell.y);
                    const uint slot = node(cell.x, cell.y);
                    const float2 w = weights(cell.x, cell.y);
                    uint edge_source = InvalidOffset;
                    if (cell.x == 0u && from.x == 0u) edge_source = h1;
                    if (cell.y == 0u && from.y == 0u) edge_source = h0;
                    if (cell.y == c + 1u - cell.x && from.y == c + 1u - from.x) edge_source = h2;
                    if (slot != InvalidOffset) ctx.WriteCorner(job, hd, v_out, loop.SourceA[slot], loop.SourceB[slot], loop.Weight[slot], edge_source, true);
                    else ctx.WriteCorner4(job, hd, v_out, uint4(h0, h1, h2, h2), w.x, w.y, edge_source, true);
                }
                TopoEmitFace(ctx, job, emitter.Face, emitter.Base, f, true);
            }
            emitter.Advance(3u);
        };
        for (uint r = 0u; r <= c; ++r) {
            const uint len = c + 2u - r;
            for (uint j = 0u; j + 1u < len; ++j) {
                triangle(uint2(r, j), uint2(r, j + 1u), uint2(r + 1u, j));
                if (j + 2u < len) triangle(uint2(r, j + 1u), uint2(r + 1u, j + 1u), uint2(r + 1u, j));
            }
        }
        return c > 1u ? (c - 1u) * c / 2u : 0u;
    }
    uint2 chords[SubdivideMaxChords];
    uint chord_count = 0u;
    if (k == 2u) {
        // The first run of cuts pairs with the second run in reverse, from the run ends nearest each other.
        const uint a = loop.CutStart;
        uint b = (a + cuts) % loop.Length;
        for (uint step = 0u; step < loop.Length && !(loop.Cut[b] && !loop.Cut[(b + loop.Length - 1u) % loop.Length]); ++step) b = (b + 1u) % loop.Length;
        const uint b_last = (b + cuts - 1u) % loop.Length;
        for (uint j = 0u; j < cuts && chord_count < SubdivideMaxChords; ++j) {
            chords[chord_count++] = uint2((a + j) % loop.Length, (b_last + loop.Length - j) % loop.Length);
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
                if (c % 2u != 0u && chord_count < SubdivideMaxChords) chords[chord_count++] = uint2(at(c - i - 1u + add), at(i + c + 1u));
                add = c * 2u + 2u;
            }
            if (chord_count < SubdivideMaxChords) chords[chord_count++] = uint2(at(c - i - 1u + add), at(i + c + 1u));
        }
        for (uint i = 0u; i < c / 2u + 1u; ++i) {
            if (chord_count < SubdivideMaxChords) chords[chord_count++] = uint2(at(i), at((c - i) + c * 2u + 1u));
        }
    }
    TopoSubdivideByChords(emitter, loop, chords, chord_count);
    return 0u;
}

// Splits a face along chords between its consecutive selected corners, skipping pairs the loop already joins.
// Returns zero, since a connect adds no interior vertices.
inline uint TopoConnectFace(TopoContext ctx, MeshTopologyJob job, uint f, thread SubdivideEmitter &emitter) {
    SubdivideLoop loop;
    TopoPlainLoop(ctx, job, f, loop, true);
    const uint2 range = ctx.SrcFaceRange(job, f);
    const uint n = loop.Length;
    device const uint *corners = ctx.SrcCorners(job);
    uint selected[SubdivideMaxCorners];
    uint selected_count = 0u;
    for (uint k = 0u; k < n; ++k) {
        if (ctx.SrcSelectedVertex(job, corners[range.x + k])) selected[selected_count++] = k;
    }
    uint2 chords[SubdivideMaxChords];
    uint chord_count = 0u;
    if (selected_count >= 2u && n > 3u) {
        for (uint i = 0u; i < selected_count && chord_count < SubdivideMaxChords; ++i) {
            const uint a = selected[i], b = selected[(i + 1u) % selected_count];
            const uint gap = (b + n - a) % n;
            // Two selected corners that are neighbors, or the only pair closing on itself, need no chord.
            if (gap <= 1u || gap == n - 1u || (selected_count == 2u && i == 1u)) continue;
            chords[chord_count++] = uint2(a, b);
        }
    }
    if (chord_count == 0u) {
        emitter.Whole(loop, ctx.SrcSelectedFace(job, f));
        return 0u;
    }
    TopoSubdivideByChords(emitter, loop, chords, chord_count);
    return 0u;
}

// Emits a face's split for the subdivide or connect operator and returns the interior vertices it adds.
inline uint TopoSplitFace(TopoContext ctx, MeshTopologyJob job, uint f, thread SubdivideEmitter &emitter) {
    return job.Op == MeshTopologyOp::Subdivide ? TopoSubdivideFace(ctx, job, f, emitter) : TopoConnectFace(ctx, job, f, emitter);
}

#endif
