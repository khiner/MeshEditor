#pragma once

#include "gpu/ConnectivityRef.h"
#include "gpu/ElementWork.h"
#include "gpu/MeshTopologyArenas.h"

// Per vertex flags of a simplify.
GPU_CONSTANT uint32_t SimplifySelected = 1u;
GPU_CONSTANT uint32_t SimplifyLocked = 2u; // On a seam, a sharp edge, a nonmanifold edge or a material boundary
GPU_CONSTANT uint32_t SimplifyBoundary = 4u;

// A plane quadric's words: ten coefficients about its three-coordinate origin.
GPU_CONSTANT uint32_t SimplifyQuadricWords = 13u;

// A simplify's state in scratch words, indexed by the record's element handles: the persistent vertex runs, and the edge runs one round of collapses writes.
struct SimplifyJob {
    uint32_t Quadrics; // Per vertex: the plane quadric over its faces and the ones its collapsed vertices held
    uint32_t Flags; // Per vertex
    uint32_t Errors; // Per edge: the collapse's cost, or infinity for none or once a winner near it blocks it
    uint32_t Positions; // Per edge: the three coordinates the survivor takes
    uint32_t Reductions; // Per edge: the triangles the collapse removes
    uint32_t Winners; // Per edge: 1 where the collapse runs this round
    uint32_t Counts; // The elected collapses, their triangle reduction, the last election pass's collapses, and the selection's triangles
    uint32_t Histogram; // SimplifyHistogramBins reduction sums over the floored cost bits
    float MaxCost; // A collapse costs below this
    float Threshold; // An elected collapse costs at most this
    float Floor; // The histogram and the trim take a cost at most this as this
    uint32_t HistogramMask, HistogramPrefix; // The histogram counts the floored cost bits whose bits under the mask equal the prefix
    uint32_t HistogramShift; // The histogram bins the bits outside the mask from this bit up
    uint32_t HistogramWinners; // 1: the histogram counts the winners instead of every proposal
    uint32_t Kept; // The trim keeps the winners whose floored cost bits are at most this
};
GPU_CONSTANT uint32_t SimplifyCountWords = 4u;
GPU_CONSTANT uint32_t SimplifyHistogramBins = 4096u;

// The simplify passes over a record's live vertices, edges or faces, one thread per element slot of the domain's blocks.
struct SimplifyPushConstants {
    SlotOffset Job; // The SimplifyJob
    DomainBlocks Blocks;
    ConnectivityRef Connectivity;
    MeshTopologyArenas Arenas;
    uint32_t CornerAttributes; // The record's corner layer bits
    SlotOffset Selected; // One bit per vertex handle
};
