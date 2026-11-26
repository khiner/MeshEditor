#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/ElementLayers.h"
#include "gpu/ElementWork.h"
#include "gpu/MeshTopologyArenas.h"

// Per vertex flags of a simplify.
GPU_CONSTANT uint32_t SimplifySelected = 1u;
GPU_CONSTANT uint32_t SimplifyLocked = 2u; // On a seam, a sharp edge, a nonmanifold edge or a material boundary
GPU_CONSTANT uint32_t SimplifyBoundary = 4u;

// A plane quadric's words: ten coefficients about its three-coordinate origin.
GPU_CONSTANT uint32_t SimplifyQuadricWords = 13u;

// A record's simplify in scratch words, whose runs are indexed by the record's element handles: the persistent vertex runs, and the edge runs one round of collapses writes.
// The fields before Threshold are the host's, which each round renews, and the rest the passes write.
struct SimplifyJob {
    DomainBlocks Vertices, Edges, Faces;
    ConnectivityRef Connectivity;
    MeshTopologyArenas Arenas;
    ElementLayers Layers; // The record's element layers
    uint32_t CornerAttributes; // The record's corner layer bits
    SlotOffset Selected; // One bit per vertex handle
    uint32_t Quadrics; // Per vertex: the plane quadric over its faces and the ones its collapsed vertices held
    uint32_t Flags; // Per vertex
    uint32_t Errors; // Per edge: the collapse's cost, or infinity for none or once a winner near it blocks it
    uint32_t Positions; // Per edge: the three coordinates the survivor takes
    uint32_t Reductions; // Per edge: the triangles the collapse removes
    uint32_t Winners; // Per edge: 1 where the collapse runs this round
    uint32_t Counts; // SimplifyCountWords
    uint32_t Histogram; // SimplifyHistogramBins reduction sums over the floored cost bits
    float MaxCost; // A collapse costs below this
    uint32_t Budget; // The triangles the round's collapses remove at most
    float Floor; // The proposals' histograms take a cost at most this as this
    float Threshold; // An elected collapse costs at most this
    uint32_t Bits, Below, At; // The select's floored cost bits so far, with the reductions below those bits and at them
    uint32_t Crossing; // The least floored cost bits at which the proposals reach the budget, or infinity's when they fall short
    uint32_t Kept; // The trim keeps the winners whose floored cost bits are at most this
    uint32_t Reduced; // The reduction of the winners the trim keeps
};
// The counts: the elected collapses, their triangle reduction, the selection's triangles, then the collapses each election pass elected, by the pass's parity.
GPU_CONSTANT uint32_t SimplifyElected = 3u;
GPU_CONSTANT uint32_t SimplifyCountWords = 5u;
GPU_CONSTANT uint32_t SimplifyHistogramBins = 4096u;

// A simplify pass's job: the record's SimplifyJob and the pass's parameter.
// A Block or Elect pass's parameter is its number in the round's election.
// A Histogram or Select pass's parameter is its level, with SimplifySelectWinners when it selects among the winners, and a trim's carries SimplifyLastParity when the election's last pass is odd.
struct SimplifyPass {
    SlotOffset Job;
    uint32_t Parameter;
};
GPU_CONSTANT uint32_t SimplifySelectWinners = 4u;
GPU_CONSTANT uint32_t SimplifyLastParity = 8u;
