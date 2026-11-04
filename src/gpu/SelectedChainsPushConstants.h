#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"

// A loop's measures: its winding by the face halfedges supplying its corners, and whether a face already spans exactly its vertices.
// Then the area normal and centroid of its vertices, and its first and last vertex positions.
struct ChainSummary {
    int32_t Winding DEFAULT();
    uint32_t MatchesFace DEFAULT();
    vec3 Normal DEFAULT();
    vec3 Centroid DEFAULT();
    vec3 First DEFAULT();
    vec3 Last DEFAULT();
};

// A listed edge's origin and end, its source halfedge, its first halfedge's face or InvalidOffset for a wire, and how many listed faces use it.
struct EdgeFacts {
    uint32_t From DEFAULT(), To DEFAULT(), Source DEFAULT(), Face DEFAULT(), SelectedFaces DEFAULT();
};

// A record's live edges, one thread per slot of its edge blocks, and the sorted selected edges.
// Oriented edge 2e runs along edge e's first halfedge and 2e + 1 against it, so a chain's two directions are two chains, of which the lower keyed one is kept.
// Each per-oriented-edge run holds one word per oriented edge below twice EdgeBound, and a doubled run holds a round's input and output halves.
struct SelectedChainsPushConstants {
    ConnectivityRef Connectivity DEFAULT();
    DomainBlocks Blocks DEFAULT(); // Edge blocks
    uint32_t VertexSlot DEFAULT(InvalidSlot);
    uint32_t EdgeBound DEFAULT();
    uint32_t ScratchSlot DEFAULT(InvalidSlot); // The buffer every run below offsets into
    uint32_t Edges DEFAULT(), EdgeCount DEFAULT();
    uint32_t BoundaryOnly DEFAULT(); // Only edges with at most one face side join the chains or supply a source
    uint32_t Faces DEFAULT(), ListedFaces DEFAULT(), Counts DEFAULT(); // The listed faces, their count and, per edge, how many of them use it
    uint32_t EdgeInfo DEFAULT(); // Per listed edge: its EdgeFacts
    uint32_t Degree DEFAULT(), Lowest DEFAULT(), Highest DEFAULT(); // Per vertex: its selected edges, and the lowest and highest of them
    uint32_t Junctions DEFAULT(); // One word, set when a vertex has more than two selected edges
    uint32_t Candidate DEFAULT(), Next DEFAULT(), Previous DEFAULT(); // Per oriented edge: whether it is live and selected, and the oriented edges after and before it in its chain
    // Per oriented edge, doubled: the jumps forward and backward, and the lowest forward key of its chain as an origin and end pair and its oriented edge.
    // Then, doubled: the lowest reverse key pair, and its chain's tail and head.
    uint32_t Jumps DEFAULT(), PreviousJumps DEFAULT(), Key DEFAULT(), KeyIndex DEFAULT(), Reverse DEFAULT(), Tails DEFAULT(), Heads DEFAULT();
    uint32_t Closed DEFAULT(), Distances DEFAULT(), Successors DEFAULT(); // Per oriented edge: whether its chain closes, then doubled its distance and successor along the chain
    uint32_t Lengths DEFAULT(), Flags DEFAULT(); // Per oriented edge then a terminator: its chain's vertex count at the representative, and one at each kept representative, each scanned in place
    uint32_t Records DEFAULT(), Output DEFAULT(); // Per chain: its first pair, its vertex count and whether it closes, then per chain vertex the vertex and the halfedge supplying the corner before it
    uint32_t ChainCount DEFAULT(), Summaries DEFAULT(); // Per chain: a ChainSummary
    uint32_t Points DEFAULT(), PointCount DEFAULT(), Angles DEFAULT(); // A point cloud's vertices and, per point then a flag, its angle about the cloud's center in its plane
    vec3 Point DEFAULT(); // The position a nearest search measures from
    uint32_t Nearest DEFAULT(); // One word: the chain vertex nearest the point
    uint32_t Parameter DEFAULT(); // A round's buffer parity
};
