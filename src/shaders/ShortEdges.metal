#include "ConnectivityRead.metal"
#include "gpu/ShortEdgesPushConstants.h"

// Sets the flag once any edge of the record is shorter than the distance.
kernel void FlagShortEdges(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant ShortEdgesPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i >= pc.HalfedgeCount) return;
    const uint h = i;
    const ConnectivityView view{b, pc.Connectivity, pc.FaceCount};
    if (view.Edge(h) == InvalidOffset || !view.EdgeFirst(h)) return;
    device const uint *corners = BindlessBuffer(uint, b.IndexBuffer, pc.CornerSlot);
    device const Vertex *vertices = BindlessBuffer(Vertex, b.VertexBuffer, pc.VertexSlot);
    const auto position = [&](uint corner) { return float3(vertices[view.VertexAddress(corners[view.HalfedgeAddress(corner)])].Position); };
    if (distance(position(h), position(view.Previous(h))) >= pc.Distance) return;
    BindlessBufferMutable(uint, b.Buffer, pc.Flag.Slot)[pc.Flag.Offset] = 1u;
}
