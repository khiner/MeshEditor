#pragma once

#include "gpu/BindlessBindings.h"
#include "gpu/SlotOffset.h"
#include "mesh/MeshPipelines.h"
#include "metal/BufferArena.h"
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

// The jobs of one recording with their tile lists per domain, laid out as ranges of a shared word arena whose one slot every pass reads.
// The scratch range begins with each domain's indirect dispatch arguments, three words per domain, that every encode refills from its tile counts.
// A job lays its scratch out from the words AllocateScratch hands it, in the order it claims them, and every recording reuses the range.
// An indirect pass binds the arena's storage directly, so the arena keeps its storage from recording that pass to the chain's submit.
template<typename Job, size_t Domains>
struct TiledJobBatch {
    static constexpr uint32_t ArgumentWords{3 * Domains};
    static_assert(sizeof(Job) % sizeof(uint32_t) == 0);

    // Reserves `scratch_words` words after the argument words for the jobs to claim.
    TiledJobBatch(BufferArena<uint32_t> &storage, uint32_t scratch_words)
        : Storage{storage}, ScratchOffset{AllocateAligned(uint64_t(scratch_words) + ArgumentWords)}, ScratchCapacity{scratch_words + ArgumentWords} {}

    // Starts a recording's job list over a fresh scratch layout.
    void Begin() {
        Jobs.clear();
        for (auto &tiles : Tiles) tiles.clear();
        ScratchWords = ArgumentWords;
    }
    // Claims `words` of scratch for the next job and returns their first word.
    uint32_t AllocateScratch(uint32_t words) {
        if (uint64_t(ScratchWords) + words > ScratchCapacity) throw std::length_error("GPU batch scratch exceeds its reservation.");
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
    // Another allocation from the storage can move them, so a caller takes the span after its last allocation.
    std::span<uint32_t> ScratchSpan() const { return Storage.GetMutable({ScratchOffset, ScratchWords}); }
    // The storage slot and the scratch's first word, for passes outside the batch that read its scratch.
    SlotOffset ScratchBinding() const { return {Storage.Buffer.Slot, ScratchOffset}; }

    // The job and per-domain tile counts of one upload, which the passes of its chain submit dispatch over.
    struct Upload {
        uint64_t Submission;
        uint32_t Jobs;
        std::array<uint32_t, Domains> Tiles;
        uint32_t JobsOffset, TileMapOffset;
    };

    // Uploads the jobs and tiles into a fresh range of the storage, refills the indirect argument words, and records the passes in order into `chain`.
    // `pc` carries the pass-specific push constants, whose storage slot and job, tile-map and scratch offsets this fills.
    // The argument words are CPU writes, so a batch uploads once per chain submit.
    // Returns the upload, which later passes of the same chain submit record over.
    template<typename PC>
    Upload Encode(mtl::ComputeChain &chain, const MeshPipelines &pipelines, PC pc, std::span<const TiledPass> passes) {
        if (RecordedSubmission == chain.Submission()) throw std::logic_error("A tiled job batch records once per chain submit.");
        RecordedSubmission = chain.Submission();
        Upload upload{.Submission = chain.Submission(), .Jobs = uint32_t(Jobs.size()), .Tiles = {}, .JobsOffset = 0u, .TileMapOffset = 0u};
        uint64_t tile_count = 0u;
        for (size_t d = 0; d < Domains; ++d) tile_count += upload.Tiles[d] = uint32_t(Tiles[d].size());
        const auto job_words = uint64_t(Jobs.size()) * (sizeof(Job) / sizeof(uint32_t));
        // Tiles start on a 16-byte boundary after the jobs, as the shaders read them as aligned pairs.
        upload.JobsOffset = AllocateAligned(job_words + 3u + 2u * tile_count);
        upload.TileMapOffset = uint32_t((upload.JobsOffset + job_words + 3u) & ~uint64_t{3u});
        std::ranges::copy(std::as_bytes(std::span{Jobs}), std::as_writable_bytes(Storage.GetMutable({upload.JobsOffset, uint32_t(job_words)})).begin());
        auto tiles = std::as_writable_bytes(Storage.GetMutable({upload.TileMapOffset, uint32_t(2u * tile_count)})).begin();
        for (const auto &domain : Tiles) tiles = std::ranges::copy(std::as_bytes(std::span{domain}), tiles).out;
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
        pc.StorageSlot = Storage.Buffer.Slot;
        pc.JobsOffset = upload.JobsOffset;
        pc.TileMapOffset = upload.TileMapOffset;
        pc.ScratchOffset = ScratchOffset;
        // Eight simd-group sums and the threadgroup total, padded to Metal's 16-byte granule.
        chain.Encode([](MTL::ComputeCommandEncoder *encoder) { encoder->setThreadgroupMemoryLength(48, 0); });
        for (const auto &pass : passes) {
            const auto groups = pass.Domain == PerJob ? upload.Jobs : upload.Tiles[pass.Domain];
            if (groups == 0) continue;
            pc.FirstTile = pass.Domain == PerJob ? 0u : first_tile[pass.Domain];
            if constexpr (requires { pc.PassParameter; }) pc.PassParameter = pass.Parameter;
            if (pass.Indirect) chain.Indirect(pipelines[pass.Pipeline], pc, Storage.Buffer, uint64_t(ScratchOffset + 3 * pass.Domain) * sizeof(uint32_t));
            else chain.Groups(pipelines[pass.Pipeline], pc, groups);
        }
    }

    std::vector<Job> Jobs;
    std::array<std::vector<uvec2>, Domains> Tiles;
    uint32_t ScratchWords{0};

private:
    // The first word of a fresh range of at least `words` words that starts on a 16-byte boundary like a buffer, so shaders read typed records there.
    uint32_t AllocateAligned(uint64_t words) {
        if (words + 3u > UINT32_MAX) throw std::length_error("GPU batch exceeds its storage's address space.");
        return (Storage.Allocate(uint32_t(words + 3u)).Offset + 3u) & ~3u;
    }

    BufferArena<uint32_t> &Storage;
    uint32_t ScratchOffset, ScratchCapacity;
    std::optional<uint64_t> RecordedSubmission;
};
