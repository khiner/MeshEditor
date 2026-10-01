#pragma once

#include "gpu/BindlessBindings.h"
#include "mesh/MeshPipelines.h"
#include "metal/Buffer.h"
#include "metal/Dispatch.h"
#include "numeric/uvec2.h"

#include <Metal/MTLComputeCommandEncoder.hpp>
#include <array>
#include <span>
#include <utility>
#include <vector>

// A pass over one tile domain, or over the jobs themselves when Domain is PerJob.
// Parameter reaches the kernel through the push constants' PassParameter when the constants declare one.
// An indirect pass takes its threadgroup count from the domain's argument words, which a kernel may zero to skip the passes after it.
struct TiledPass {
    MeshPass Pipeline;
    uint32_t Domain;
    uint32_t Parameter{};
    bool Indirect{false};
};
// Dispatches one threadgroup per job, which reads its job by threadgroup rather than by tile.
constexpr uint32_t PerJob{~0u};

// The jobs of one recording with their tile lists per domain, over scratch, job, and tile buffers reused across recordings.
// Scratch begins with each domain's indirect dispatch arguments, three words per domain, that every encode refills from its tile counts.
// A job lays its scratch out from the words AllocateScratch hands it, in the order it claims them.
template<typename Job, size_t Domains>
struct TiledJobBatch {
    static constexpr uint32_t ArgumentWords{3 * Domains};

    TiledJobBatch(mtl::BufferContext &ctx, uint32_t widest_scratch_words, uint32_t most_jobs)
        : Scratch{ctx, 0u, SlotType::Buffer,mtl::BufferLifetime::Workspace},
          JobBuffer{ctx, uint64_t(most_jobs) * sizeof(Job), SlotType::Buffer,mtl::BufferLifetime::Workspace},
          TileBuffer{ctx, (uint64_t(widest_scratch_words) / 32u + uint64_t(most_jobs) * 8u) * sizeof(uvec2), SlotType::Buffer,mtl::BufferLifetime::Workspace} {
        Scratch.SetUsedSize((uint64_t(widest_scratch_words) + ArgumentWords) * sizeof(uint32_t));
    }

    // Starts a recording's job list over a fresh scratch layout.
    void Begin() {
        Jobs.clear();
        for (auto &tiles : Tiles) tiles.clear();
        ScratchWords = ArgumentWords;
    }
    // Claims `words` of scratch for the next job and returns their first word.
    uint32_t AllocateScratch(uint32_t words) {
        if (uint64_t(ScratchWords) + words > UINT32_MAX) throw std::length_error("GPU batch scratch exceeds its address space.");
        return std::exchange(ScratchWords, ScratchWords + words);
    }
    // Adds a job covering `tile_counts[domain]` tiles per domain and returns its index.
    uint32_t AddJob(const Job &job, std::array<uint32_t, Domains> tile_counts) {
        const auto index = uint32_t(Jobs.size());
        for (size_t d = 0; d < Domains; ++d) {
            for (uint32_t t = 0; t < tile_counts[d]; ++t) Tiles[d].emplace_back(index, t);
        }
        Jobs.push_back(job);
        return index;
    }
    // Replaces a domain's tiles with `tile_counts[job]` per job, for a domain an earlier submit's results size.
    void SetDomainTiles(uint32_t domain, std::span<const uint32_t> tile_counts) {
        Tiles[domain].clear();
        for (uint32_t job = 0; job < tile_counts.size(); ++job) {
            for (uint32_t t = 0; t < tile_counts[job]; ++t) Tiles[domain].emplace_back(job, t);
        }
    }
    // A domain's indirect argument words, zero once a kernel has skipped its remaining indirect passes.
    uint32_t IndirectGroups(uint32_t domain) const { return ScratchSpan()[3 * domain]; }
    // The recording's scratch words, for staging inputs before the passes are recorded and reading results after they complete.
    std::span<uint32_t> ScratchSpan() const { return Scratch.GetMutableSpan<uint32_t>({0, ScratchWords}); }

    // The job and per-domain tile counts of one upload, which the passes of its chain submit dispatch over.
    struct Upload {
        uint64_t Submission;
        uint32_t Jobs;
        std::array<uint32_t, Domains> Tiles;
    };

    // Uploads the jobs and tiles, refills the indirect argument words, and records the passes in order into `chain`.
    // `pc` carries the pass-specific push constants, whose job, tile-map, and scratch slots this fills.
    // The uploads are CPU writes, so a batch uploads once per chain submit.
    // Returns the upload, which later passes of the same chain submit record over.
    template<typename PC>
    Upload Encode(mtl::ComputeChain &chain, const MeshPipelines &pipelines, PC pc, std::span<const TiledPass> passes) {
        if (RecordedSubmission == chain.Submission()) throw std::logic_error("A tiled job batch records once per chain submit.");
        RecordedSubmission = chain.Submission();
        Upload upload{.Submission = chain.Submission(), .Jobs = uint32_t(Jobs.size()), .Tiles = {}};
        std::vector<uvec2> tiles;
        for (size_t d = 0; d < Domains; ++d) {
            upload.Tiles[d] = uint32_t(Tiles[d].size());
            tiles.insert(tiles.end(), Tiles[d].begin(), Tiles[d].end());
        }
        JobBuffer.Update(as_bytes(Jobs));
        TileBuffer.Update(as_bytes(tiles));
        const auto arguments = ScratchSpan();
        for (size_t d = 0; d < Domains; ++d) {
            arguments[3 * d] = upload.Tiles[d];
            arguments[3 * d + 1] = arguments[3 * d + 2] = 1u;
        }
        Record(chain, pipelines, upload, pc, passes);
        return upload;
    }
    // Records more passes over the jobs and tiles of `upload`, which this chain submit's Encode returned.
    template<typename PC>
    void Record(mtl::ComputeChain &chain, const MeshPipelines &pipelines, const Upload &upload, PC pc, std::span<const TiledPass> passes) {
        if (upload.Submission != chain.Submission()) throw std::logic_error("A tiled job batch records over its upload in the same chain submit.");
        std::array<uint32_t, Domains> first_tile{};
        for (size_t d = 1; d < Domains; ++d) first_tile[d] = first_tile[d - 1] + upload.Tiles[d - 1];
        pc.JobsSlot = JobBuffer.Slot;
        pc.TileMapSlot = TileBuffer.Slot;
        pc.ScratchSlot = Scratch.Slot;
        // Eight simd-group sums and the threadgroup total, padded to Metal's 16-byte granule.
        chain.Encoder()->setThreadgroupMemoryLength(48, 0);
        for (const auto &pass : passes) {
            const auto groups = pass.Domain == PerJob ? upload.Jobs : upload.Tiles[pass.Domain];
            if (groups == 0) continue;
            pc.FirstTile = pass.Domain == PerJob ? 0u : first_tile[pass.Domain];
            if constexpr (requires { pc.PassParameter; }) pc.PassParameter = pass.Parameter;
            if (pass.Indirect) chain.Indirect(pipelines[pass.Pipeline], pc, Scratch, uint64_t(3 * pass.Domain) * sizeof(uint32_t));
            else chain.Groups(pipelines[pass.Pipeline], pc, groups);
        }
    }

    std::vector<Job> Jobs;
    std::array<std::vector<uvec2>, Domains> Tiles;
    uint32_t ScratchWords{0};
    mtl::Buffer Scratch, JobBuffer, TileBuffer;

private:
    std::optional<uint64_t> RecordedSubmission;
};
