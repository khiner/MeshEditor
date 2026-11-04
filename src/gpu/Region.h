#pragma once
#include "gpu/Collider.h"
#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"

// Which faces of the arrangement a region fill keeps, by its operands' winding numbers.
// In enum order, the fills keep the first's positive, either positive, the first's positive without the second's, or both positive.
enum class RegionFill : uint32_t { Positive,
                                   Union,
                                   Difference,
                                   Intersection };

// A directed segment of an operand's face loops, with the operand's material on its left.
struct RegionSegment {
    uint32_t Start DEFAULT(), End DEFAULT(), Operand DEFAULT();
};

// A segment's crossing with another, at Position and Parameter along this segment.
// Delta is the change of the other operand's winding on this segment's right past the crossing.
// Shift is the parameter's first-order move, which orders crossings whose parameters tie.
struct RegionCrossing {
    uint32_t Other DEFAULT();
    int32_t Delta DEFAULT();
    vec2 Position DEFAULT();
    float Parameter DEFAULT(), Shift DEFAULT();
};

GPU_CONSTANT uint32_t RegionIncidentEnd = 0x80000000u; // An incident entry whose vertex is the segment's end

// Scratch word offsets of a region fill.
// Vertices and segments number the first operand's before the second's.
struct RegionJob {
    RegionFill Fill DEFAULT();
    GpuArray<uint32_t, 2> Vertices DEFAULT(), Segments DEFAULT();
    uint32_t Positions DEFAULT(), Moves DEFAULT(); // Per vertex: packed_float2, its plane position and the displacement deciding its ties
    uint32_t Bounds DEFAULT(); // Per operand: the ordered bits of its positions' minimum x and y then maximum x and y
    uint32_t SegmentRecords DEFAULT(); // Per segment: RegionSegment
    // Per vertex then a terminator: its first incident segment end, with the fill's cursors per vertex.
    // Then per segment end: the segment, with RegionIncidentEnd where the vertex is its end.
    uint32_t IncidentFirst DEFAULT(), IncidentCursor DEFAULT(), Incident DEFAULT();
    ColliderRuns Collider DEFAULT(); // Over every segment
    // Per segment then a terminator: its first crossing entry, then per entry a RegionCrossing in order along its segment.
    uint32_t CrossingFirst DEFAULT(), Crossings DEFAULT();
    uint32_t Windings DEFAULT(); // Per vertex: its two operand windings in the moved configuration
    // A segment's pieces run between its ends and its crossings, piece k of segment s at CrossingFirst[s] + s + k.
    // Per piece then a terminator: whether its sides differ in fill, scanned in place to its kept index, and per piece whether the fill is on its right.
    uint32_t Pieces DEFAULT(), PieceKept DEFAULT(), PieceReversed DEFAULT();
    // Per kept piece: its piece and the kept piece after it around its loop.
    // Then two buffers each of its loop label and jump, and of its distance and successor along the loop.
    uint32_t KeptPieces DEFAULT(), Kept DEFAULT(), Next DEFAULT(), Labels DEFAULT(), Jumps DEFAULT(), Distances DEFAULT(), Successors DEFAULT();
    // Per kept piece then a terminator: its loop's length at its representative, and one at each representative, each scanned in place.
    uint32_t LoopLengths DEFAULT(), LoopFlags DEFAULT();
    // Per loop: its first output point and its length, then per kept piece its oriented start point, in loop order.
    uint32_t LoopRecords DEFAULT(), Output DEFAULT();
    uint32_t Fault DEFAULT(); // One word, set when a kept piece finds no kept piece after it, which a degenerate arrangement leaves
};

// One operand's segments: its record's face loop edges, or its silhouette along the plane normal, moved into the plane's space and projected onto it.
struct RegionSegmentsPushConstants {
    ConnectivityRef Connectivity DEFAULT();
    uint32_t VertexSlot DEFAULT(InvalidSlot), FaceNormalSlot DEFAULT(InvalidSlot);
    ElementWork Faces DEFAULT();
    uint32_t FaceCount DEFAULT();
    uint32_t Operand DEFAULT(), VertexBase DEFAULT(), SegmentBase DEFAULT(); // The operand and its first vertex and segment in the job
    uint32_t Cusps DEFAULT(); // Only halfedges whose face faces the normal and whose opposite face does not
    mat3 Rotation DEFAULT();
    vec3 Translation DEFAULT();
    vec3 Point DEFAULT(), Normal DEFAULT(), U DEFAULT(), V DEFAULT(); // The plane and the basis of its coordinates
    SlotOffset Aliases DEFAULT(); // Per vertex of the job: the vertex it merges into, or itself, which a zero-length segment's ends join through
    SlotOffset Counts DEFAULT(); // Per face then a terminator: its emitted segments, scanned to its first slot
    SlotOffset Positions DEFAULT(), Segments DEFAULT(), Bounds DEFAULT();
    SlotOffset Deviation DEFAULT(); // One word: the ordered bits of the farthest face edge end from the plane
};
