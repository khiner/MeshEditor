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

// One record's job of a field over the live elements of its domain that reads their neighbors, one thread per element slot of its blocks.
struct NeighborFieldJob {
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
};
// One record's job writing a field over the live elements of its domain from its connectivity and positions alone, one thread per element slot of its blocks.
// Plane Side reads the plane's unit normal and offset, the tolerance relative to the larger of one and a vertex's distance from the origin, and the selected side.
struct AttributeFieldJob {
    DomainBlocks Blocks;
    ConnectivityRef Connectivity;
    uint32_t VertexSlot;
    Element Domain;
    NamedAttributeRef Output;
    vec3 PlaneNormal;
    float PlaneOffset, Tolerance;
    PlaneSideKind Side;
};

// How a conversion reduces the values of the elements related to a target element.
// Any and All reduce booleans, and Mean, Min and Max reduce numbers, componentwise for a vector.
// An integer mean rounds to the nearest integer, halves away from zero.
// A target element with no related element takes zero.
enum class DomainReduction : uint32_t {
    Any,
    All,
    Mean,
    Min,
    Max,
};
// One record's job reducing a stored field's values from its domain onto each live element of another domain, one thread per element slot of the target domain's blocks.
struct DomainConversionJob {
    DomainBlocks Blocks; // The record's blocks of the target domain
    ConnectivityRef Connectivity;
    NamedAttributeRef Source, Output;
    AttributeType Type;
    AttributeDomain From, To;
    DomainReduction Reduction;
};

// One record's job copying a bool attribute's words at each listed element block of its record to the block's mask block, one thread per word.
// Mask block b covers element block b at word Masks.Offset + 8b.
// An absent slot denotes an attribute absent from that domain, whose mask blocks take zero.
struct AttributeMaskJob {
    NamedAttributeRef Attribute;
    uint32_t MembershipSlot; // The domain's block metadata
    SlotOffset Blocks, Masks; // The listed element blocks, one word each
    uint32_t Words; // Eight per listed block
};
