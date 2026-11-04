#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/Element.h"
#include "gpu/ElementWork.h"
#include "gpu/NamedAttributeRef.h"
#include "gpu/PlaneSide.h"

enum class FieldAdjacency : uint32_t {
    Domain, // Vertices across edges, edges at shared vertices, faces across their opposite halfedges
    EdgeLoop,
    EdgeRing,
    Radial, // Faces sharing an edge, however many and however wound
};

// A field over the live elements of a record's domain, one thread per element slot of its blocks.
struct AttributeFieldPushConstants {
    DomainBlocks Blocks;
    ConnectivityRef Connectivity;
    uint32_t VertexSlot, FaceTriangleStartSlot, TriangleSlot;
    Element Domain;
    FieldAdjacency Adjacency;
    uint32_t Mean; // Curvature: the mean curvature instead of the Gaussian
    uint32_t Complement;
    NamedAttributeRef Seed, Output;
    // Per element handle: a component label and whether its root holds a seed, or the flags a growth step reads and writes.
    SlotOffset Labels, Flags;
    SlotOffset Changed; // One word a link round sets when a label falls
    SlotOffset Restrict; // Per element handle: whether a link round takes the element, marked from List, or no slot to take every live element
    SlotOffset List;
    uint32_t ListCount;
    uint32_t Parity; // Labels shift left and carry in the low bit the parity of the path to their root, which each adjacency step flips
    // Plane Side: the plane's unit normal and offset, the tolerance relative to the larger of one and a vertex's distance from the origin, and the selected side.
    vec3 PlaneNormal;
    float PlaneOffset, Tolerance;
    PlaneSideKind Side;
};
