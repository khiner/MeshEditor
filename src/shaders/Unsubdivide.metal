#include "ConnectivityRead.metal"
#include "gpu/UnsubdividePushConstants.h"

// A multiset hash with vertex `v` added, holding the count, the sum and the xor of the vertices' mixes.
inline uint3 UnsubdivideHashed(uint3 hash, uint v) {
    const uint m = HashMix(v);
    return uint3(hash.x + 1u, hash.y + m, hash.z ^ m);
}

struct UnsubdivideContext {
    device const BindlessSet &B;
    constant UnsubdividePushConstants &Pc;
    ConnectivityView View;
    UnsubdivideContext(device const BindlessSet &b, constant UnsubdividePushConstants &pc) : B(b), Pc(pc), View{b, pc.Connectivity} {}
    device uint *Run(uint offset) const { return BindlessBufferMutable(uint, B.Buffer, Pc.ScratchSlot) + offset; }
    uint Selected(uint i) const { return Run(Pc.Selection)[i]; }
    uint State(uint v) const { return Run(Pc.State)[v]; }
    bool IsRemoved(uint v) const { return State(v) == UnsubdivideStateRemoved; }
    device uint *Polygon(uint i) const { return Run(Pc.Polygons) + UnsubdividePolygonWords * i; }
    uint Owner(uint v) const {
        uint owner = InvalidOffset;
        for (const auto item : View.Fan(v)) owner = min(owner, item.y);
        return owner;
    }
    // The multiset hash of a face's vertices without one of them, then the lowest of those vertices.
    uint4 FaceHash(uint f, uint without) const {
        uint3 hash = uint3(0u);
        uint lowest = InvalidOffset;
        const uint2 range = View.FaceHalfedges(f);
        for (uint h = range.x; h < range.y; ++h) {
            const uint v = View.VertexOf(h);
            if (v == without) continue;
            hash = UnsubdivideHashed(hash, v);
            lowest = min(lowest, v);
        }
        return uint4(hash, lowest);
    }
    bool Exists(uint3 hash, uint at, uint excluded) const {
        for (const auto item : View.Fan(at)) {
            const uint f = item.y;
            if (f == InvalidOffset || View.Valence(f) != hash.x) continue;
            bool incident = false;
            for (const auto other : View.Fan(excluded)) incident = incident || other.y == f;
            if (!incident && all(FaceHash(f, InvalidOffset).xyz == hash)) return true;
        }
        return false;
    }
    static uint2 Key(uint3 hash) {
        const uint key = HashMix(hash.y ^ HashMix(hash.x)) ^ hash.z;
        return uint2(key == InvalidOffset ? 0u : key, HashMix(hash.z + 0x9e3779b9u) ^ hash.y);
    }
    KeyTable Table() const { return {Run(Pc.Table), Pc.TableMask}; }
    // Claims the first free slot along the key's probe for the polygon at `id`, writing the check word and the id after the claim.
    void Claim(uint3 hash, uint id) const {
        const uint2 key = Key(hash);
        Table().Claim(key, HashMix(key.x))[2] = id;
    }
    // Whether an earlier claimant along the probe holds the same key and check word, once every claim has written them.
    bool Duplicated(uint3 hash, uint id) const {
        const uint2 key = Key(hash);
        return Table().Find(key, HashMix(key.x))[2] != id;
    }
    device const uint *PolygonOf(uint v) const { return Polygon(LowerBound(Run(Pc.Selection), Pc.SelectionCount, v)); }
    // Calls visit(hash, offset, polygon) for each polygon of a touched face's record in list order.
    // The residual comes first when three corners survive, then the fans the face owns in ascending vertex order.
    // Each offset lies within the record after the face and count words.
    // `polygon` is the fan's planned polygon, or null for the residual.
    template<typename Visit> void ForEachPolygon(uint f, Visit visit) const {
        const uint2 range = View.FaceHalfedges(f);
        uint offset = 2u, surviving = 0u;
        uint3 hash = uint3(0u);
        for (uint h = range.x; h < range.y; ++h) {
            const uint v = View.VertexOf(h);
            if (IsRemoved(v)) continue;
            ++surviving;
            hash = UnsubdivideHashed(hash, v);
        }
        if (surviving >= 3u) {
            visit(hash, offset, nullptr);
            offset += 1u + 2u * surviving;
        }
        uint written = InvalidOffset;
        for (;;) {
            uint next = InvalidOffset;
            for (uint h = range.x; h < range.y; ++h) {
                const uint v = View.VertexOf(h);
                if (!IsRemoved(v) || Run(Pc.Chain)[v] || Owner(v) != f) continue;
                if ((written == InvalidOffset || v > written) && v < next) next = v;
            }
            if (next == InvalidOffset) return;
            written = next;
            device const uint *polygon = PolygonOf(next);
            visit(uint3(polygon[0], polygon[9], polygon[10]), offset, polygon);
            offset += 1u + 2u * polygon[0];
        }
    }
};

// A selected vertex is eligible with at most four edges, none a loop, every edge with one or two faces, and a fan of at most four corners.
kernel void UnsubdivideEligible(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant UnsubdividePushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.SelectionCount) return;
    const UnsubdivideContext ctx{b, pc};
    const uint v = ctx.Selected(i);
    ctx.Run(pc.Eligible)[i] = 0u;
    const uint fan = ctx.View.Incoming(v).y;
    if (fan > 4u) return;
    uint degree = 0u, boundary = 0u, manifold = 0u;
    bool valid = true;
    ctx.View.ForEachIncidentEdge(v, [&](uint e) {
        const uint h = ctx.View.EdgeHalfedge(e), a = ctx.View.From(h), c = ctx.View.VertexOf(h);
        if (degree == 4u || a == c) valid = false;
        ++degree;
        if (pc.FaceCount) {
            const uint sides = ctx.View.FaceSides(a, c);
            if (sides == 1u) ++boundary;
            else if (sides == 2u) ++manifold;
            else valid = false;
        }
    });
    if (!valid) return;
    bool chain;
    if (!pc.FaceCount) {
        if (degree != 2u) return;
        chain = true;
    } else {
        chain = degree == 2u && manifold == 2u;
        if (!chain && !((degree == 3u || degree == 4u) && manifold == degree) && !(degree == 3u && boundary == 2u && manifold == 1u)) return;
        if (chain) {
            for (const auto item : ctx.View.Fan(v)) {
                if (item.y != InvalidOffset && ctx.View.Valence(item.y) <= 3u) return;
            }
        }
    }
    ctx.Run(pc.Eligible)[i] = 1u | (chain ? 2u : 0u);
}

// Each vertex of the removal wave plans the polygon its fan collapses to, chaining the far edges of its faces into a loop or a path closed across a boundary.
kernel void UnsubdividePolygon(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant UnsubdividePushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.SelectionCount) return;
    const UnsubdivideContext ctx{b, pc};
    const uint v = ctx.Selected(i), eligible = ctx.Run(pc.Eligible)[i];
    device uint *state = ctx.Run(pc.State) + v;
    ctx.Run(pc.Chain)[v] = (eligible & 2u) ? 1u : 0u;
    if (!(eligible & 1u) || ctx.Run(pc.Tags)[i] == 0u) return;
    if (!pc.FaceCount || (eligible & 2u)) {
        *state = UnsubdivideStateCandidate;
        return;
    }
    // Ears are keyed by origin, and the far edge of each face runs from the vertex after v to the vertex before it.
    uint from[4], to[4], source[4], edge[4], faces[4];
    uint ears = 0u;
    for (const auto item : ctx.View.Fan(v)) {
        const uint h = item.x, f = item.y;
        const uint previous = ctx.View.Previous(h), next = ctx.View.Next(h, f);
        const uint a = ctx.View.VertexOf(next), c = ctx.View.VertexOf(previous);
        if (a == c) return;
        for (uint k = 0u; k < ears; ++k) {
            if (from[k] == a || to[k] == c) return;
        }
        from[ears] = a;
        to[ears] = c;
        source[ears] = previous;
        edge[ears] = ctx.View.Valence(f) == 3u ? previous : h;
        faces[ears] = f;
        ++ears;
    }
    if (!ears) return;
    uint start = InvalidOffset, open = InvalidOffset;
    for (uint k = 0u; k < ears; ++k) {
        bool arrives = false;
        for (uint j = 0u; j < ears; ++j) arrives = arrives || to[j] == from[k];
        start = min(start, from[k]);
        if (!arrives) open = min(open, from[k]);
    }
    const bool boundary = open != InvalidOffset;
    if (boundary) start = open;
    uint polygon_source[5], polygon_edge[5], polygon_vertex[5];
    uint count = 0u, at = start;
    do {
        uint found = InvalidOffset;
        for (uint k = 0u; k < ears; ++k) {
            if (from[k] == at) found = k;
        }
        if (found == InvalidOffset) break;
        polygon_source[count] = source[found];
        polygon_edge[count] = edge[found];
        polygon_vertex[count] = at;
        ++count;
        at = to[found];
    } while (at != start && count <= ears);
    if (count != ears || (!boundary && at != start)) return;
    if (boundary) {
        if (ctx.View.EdgeBetween(start, at) != InvalidOffset) return;
        uint closing = InvalidOffset;
        for (const auto item : ctx.View.Fan(v)) {
            const uint next = ctx.View.Next(item.x, item.y);
            if (ctx.View.VertexOf(next) == start) closing = next;
        }
        polygon_source[count] = polygon_edge[count] = closing;
        polygon_vertex[count] = at;
        ++count;
    }
    if (count < 3u || count > 4u) return;
    uint3 hash = uint3(0u);
    uint lowest = InvalidOffset;
    for (uint k = 0u; k < count; ++k) {
        for (uint j = 0u; j < k; ++j) {
            if (polygon_vertex[j] == polygon_vertex[k]) return;
        }
        hash = UnsubdivideHashed(hash, polygon_vertex[k]);
        lowest = min(lowest, polygon_vertex[k]);
    }
    if (ctx.Exists(hash, lowest, v)) return;
    uint3 residuals[4];
    uint residual_count = 0u;
    for (uint k = 0u; k < ears; ++k) {
        if (ctx.View.Valence(faces[k]) < 4u) continue;
        const uint4 face_hash = ctx.FaceHash(faces[k], v);
        const uint3 residual = face_hash.xyz;
        if (all(residual == hash)) return;
        for (uint j = 0u; j < residual_count; ++j) {
            if (all(residuals[j] == residual)) return;
        }
        if (ctx.Exists(residual, face_hash.w, v)) return;
        residuals[residual_count++] = residual;
    }
    device uint *polygon = ctx.Polygon(i);
    polygon[0] = count;
    for (uint k = 0u; k < 4u; ++k) {
        polygon[1u + 2u * k] = k < count ? polygon_source[k] : InvalidOffset;
        polygon[2u + 2u * k] = k < count ? polygon_edge[k] : InvalidOffset;
    }
    polygon[9] = hash.y;
    polygon[10] = hash.z;
    *state = UnsubdivideStateCandidate;
}

// A candidate decides once every lower candidate among its neighbors has decided.
kernel void UnsubdivideDecide(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant UnsubdividePushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.SelectionCount) return;
    const UnsubdivideContext ctx{b, pc};
    const uint v = ctx.Selected(i);
    if (ctx.State(v) != UnsubdivideStateCandidate) return;
    bool waiting = false, blocked = false;
    ctx.View.ForEachIncidentEdge(v, [&](uint e) {
        const uint u = ctx.View.OtherEnd(ctx.View.EdgeHalfedge(e), v);
        if (u >= v) return;
        const uint state = ctx.State(u);
        waiting = waiting || state == UnsubdivideStateCandidate;
        blocked = blocked || state == UnsubdivideStateRemoved;
    });
    if (waiting && !blocked) return;
    ctx.Run(pc.State)[v] = blocked ? UnsubdivideStateNone : UnsubdivideStateRemoved;
    ctx.Run(pc.Changed)[0] = 1u;
}

kernel void UnsubdivideCount(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant UnsubdividePushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i > pc.SelectionCount) return;
    const UnsubdivideContext ctx{b, pc};
    if (i == pc.SelectionCount) {
        ctx.Run(pc.Removed)[i] = 0u;
        return;
    }
    const uint v = ctx.Selected(i);
    const bool removed = ctx.IsRemoved(v);
    ctx.Run(pc.Removed)[i] = removed ? 1u : 0u;
    if (!removed || !pc.FaceCount) return;
    for (const auto item : ctx.View.Fan(v)) {
        if (item.y != InvalidOffset) atomic_fetch_or_explicit(AtomicWord(ctx.Run(pc.FaceWords) + item.y), 1u, memory_order_relaxed);
    }
}
// Each touched face sizes its record, which holds the face, its polygon count, then each polygon's corner count and corner and edge pairs.
kernel void UnsubdivideFaceWords(uint f [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant UnsubdividePushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (f >= pc.FaceBound) return;
    const UnsubdivideContext ctx{b, pc};
    device uint *words = ctx.Run(pc.FaceWords) + f;
    if (!*words) return;
    ctx.Run(pc.FaceFlags)[f] = 1u;
    uint length = 2u;
    ctx.ForEachPolygon(f, [&](uint3 hash, uint offset, device const uint *) { length = offset + 1u + 2u * hash.x; });
    *words = length;
}
kernel void UnsubdivideInsert(uint f [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant UnsubdividePushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (f >= pc.FaceBound) return;
    const UnsubdivideContext ctx{b, pc};
    device const uint *words = ctx.Run(pc.FaceWords);
    if (words[f + 1u] == words[f]) return;
    ctx.ForEachPolygon(f, [&](uint3 hash, uint offset, device const uint *) { ctx.Claim(hash, words[f] + offset); });
}
kernel void UnsubdivideVerify(uint f [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant UnsubdividePushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (f >= pc.FaceBound) return;
    const UnsubdivideContext ctx{b, pc};
    device const uint *words = ctx.Run(pc.FaceWords);
    if (words[f + 1u] == words[f]) return;
    ctx.ForEachPolygon(f, [&](uint3 hash, uint offset, device const uint *) {
        if (ctx.Duplicated(hash, words[f] + offset)) ctx.Run(pc.Duplicate)[0] = 1u;
    });
}

// The list holds the removed count and vertices, the touched face count and each face's record offset, then the records.
// Parameter holds the touched face count.
kernel void UnsubdivideList(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant UnsubdividePushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const UnsubdivideContext ctx{b, pc};
    device uint *list = ctx.Run(pc.List);
    device const uint *removed = ctx.Run(pc.Removed), *face_words = ctx.Run(pc.FaceWords);
    const uint removed_count = removed[pc.SelectionCount], faces = pc.Parameter;
    if (i < pc.SelectionCount) {
        if (removed[i + 1u] != removed[i]) list[1u + removed[i]] = ctx.Selected(i);
        if (i == 0u) {
            list[0] = removed_count;
            list[1u + removed_count] = faces;
        }
    }
    if (!pc.FaceCount) return;
    const uint f = i;
    if (f >= pc.FaceBound || face_words[f + 1u] == face_words[f]) return;
    const uint records = 2u + removed_count + faces;
    list[2u + removed_count + ctx.Run(pc.FaceFlags)[f]] = records + face_words[f];
    device uint *record = list + records + face_words[f];
    record[0] = f;
    uint polygons = 0u;
    const uint2 range = ctx.View.FaceHalfedges(f);
    ctx.ForEachPolygon(f, [&](uint3 hash, uint offset, device const uint *polygon) {
        device uint *out = record + offset;
        *out++ = hash.x;
        if (polygon) {
            for (uint k = 0u; k < hash.x; ++k) {
                *out++ = polygon[1u + 2u * k];
                *out++ = polygon[2u + 2u * k];
            }
        } else {
            for (uint h = range.x; h < range.y; ++h) {
                if (ctx.IsRemoved(ctx.View.VertexOf(h))) continue;
                *out++ = h;
                *out++ = h;
            }
        }
        ++polygons;
    });
    record[1] = polygons;
}
