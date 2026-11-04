#include "ConnectivityRead.metal"
#include "gpu/EditVisibilityPushConstants.h"

struct VisibilityContext {
    device const BindlessSet &B;
    constant EditVisibilityPushConstants &Pc;
    ConnectivityView View;
    VisibilityContext(device const BindlessSet &b, constant EditVisibilityPushConstants &pc) : B(b), Pc(pc), View{b, pc.Connectivity} {}
    device uint *Run(uint offset) const { return BindlessBufferMutable(uint, B.Buffer, Pc.ScratchSlot) + offset; }
    uint Handle(uint domain, uint i) const { return BlockHandle(i, B, Pc.Blocks[domain]); }
    uint Address(uint domain, uint handle) const { return domain == 0u ? View.VertexAddress(handle) : domain == 1u ? View.EdgeAddress(handle) : View.FaceAddress(handle); }
    bool Masked(uint slot, uint address) const { return MaskBit(BindlessBuffer(uint, B.Buffer, slot), address); }
    bool Hidden(uint domain, uint handle) const { return Masked(Pc.HiddenSlot[domain], Address(domain, handle)); }
    bool Selected(uint domain, uint handle) const { return Masked(Pc.SelectionSlot[domain], Address(domain, handle)); }
    bool Changed(uint domain, uint handle) const { return MaskBit(Run(Pc.Changes[domain]), handle); }
    bool HiddenAfter(uint domain, uint handle) const { return Hidden(domain, handle) || Changed(domain, handle); }
    void Mark(uint domain, uint handle) const { atomic_fetch_or_explicit(AtomicWord(Run(Pc.Changes[domain]) + handle / 32u), 1u << (handle % 32u), memory_order_relaxed); }
    void Hide(uint domain, uint handle) const { if (!Hidden(domain, handle)) Mark(domain, handle); }
    template<typename Visit> void ForEachEdgeFace(uint e, Visit visit) const {
        const uint h = View.EdgeHalfedge(e), c = View.VertexOf(h);
        for (const auto item : View.Fan(View.From(h))) {
            if (View.FanSide(item, c) != InvalidOffset) visit(item.y);
        }
    }
    bool Isolated(uint v) const {
        bool isolated = true;
        View.ForEachIncidentEdge(v, [&](uint e) { isolated &= HiddenAfter(1u, e); });
        return isolated;
    }
};

// The source domain's visible selected elements hide, or its visible unselected ones.
kernel void EditVisibilitySource(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant EditVisibilityPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const VisibilityContext ctx{b, pc};
    const uint handle = ctx.Handle(pc.Source, i);
    if (handle == InvalidOffset || ctx.Hidden(pc.Source, handle)) return;
    if (ctx.Selected(pc.Source, handle) != (pc.Unselected != 0u)) ctx.Hide(pc.Source, handle);
}
kernel void EditVisibilityReveal(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant EditVisibilityPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const VisibilityContext ctx{b, pc};
    const uint handle = ctx.Handle(pc.Source, i);
    if (handle == InvalidOffset || !ctx.Hidden(pc.Source, handle)) return;
    ctx.Mark(pc.Source, handle);
}
// Hiding faces hides each edge of a newly hidden face whose faces are all hidden.
// Hiding the unselected faces of a mesh without faces hides its unselected edges.
kernel void EditVisibilityEdges(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant EditVisibilityPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const VisibilityContext ctx{b, pc};
    const uint e = ctx.Handle(1u, i);
    if (e == InvalidOffset || ctx.Hidden(1u, e)) return;
    const uint h = ctx.View.EdgeHalfedge(e);
    bool hide = false;
    if (pc.Source == 0u) {
        hide = ctx.Changed(0u, ctx.View.From(h)) || ctx.Changed(0u, ctx.View.VertexOf(h));
    } else if (pc.Source == 2u) {
        bool reached = false, visible = false;
        ctx.ForEachEdgeFace(e, [&](uint f) {
            reached |= ctx.Changed(2u, f);
            visible |= !ctx.HiddenAfter(2u, f);
        });
        hide = (reached && !visible) || (pc.Unselected != 0u && pc.FaceCount == 0u && !ctx.Selected(1u, e));
    }
    if (hide) ctx.Hide(1u, e);
}
kernel void EditVisibilityFaces(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant EditVisibilityPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const VisibilityContext ctx{b, pc};
    const uint f = ctx.Handle(2u, i);
    if (f == InvalidOffset || ctx.Hidden(2u, f)) return;
    const uint2 range = ctx.View.FaceHalfedges(f);
    bool hide = false;
    for (uint h = range.x; h < range.y && !hide; ++h) hide = pc.Source == 0u ? ctx.Changed(0u, ctx.View.VertexOf(h)) : ctx.Changed(1u, ctx.View.Edge(h));
    if (hide) ctx.Hide(2u, f);
}
// A vertex of a newly hidden edge or face hides once every edge at it is hidden, as does an unselected vertex when hiding the unselected edges or faces.
kernel void EditVisibilityVertices(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant EditVisibilityPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    const VisibilityContext ctx{b, pc};
    const uint v = ctx.Handle(0u, i);
    if (v == InvalidOffset || ctx.Hidden(0u, v)) return;
    bool reached = pc.Unselected != 0u && !ctx.Selected(0u, v);
    if (!reached) {
        if (pc.Source == 1u) ctx.View.ForEachIncidentEdge(v, [&](uint e) { reached |= ctx.Changed(1u, e); });
        else {
            for (const auto item : ctx.View.Fan(v)) reached |= item.y != InvalidOffset && ctx.Changed(2u, item.y);
        }
    }
    if (reached && ctx.Isolated(v)) ctx.Hide(0u, v);
}
