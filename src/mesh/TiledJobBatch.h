#pragma once

#include "numeric/uvec2.h"

#include "gpu/BindlessBindings.h"
#include "gpu/Types.h"
#include "mesh/MeshPipelines.h"
#include "metal/Bindless.h"
#include "metal/Buffer.h"
#include "metal/MetalContext.h"

#include <Metal/MTLCommandBuffer.hpp>
#include <Metal/MTLCommandQueue.hpp>
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

// The jobs of one submit with their tile lists per domain, over scratch, job, and tile buffers reused across submits.
// Scratch begins with each domain's indirect dispatch arguments, three words per domain, that every encode refills from its tile counts.
// A job lays its scratch out from the words AllocateScratch hands it, in the order it claims them.
template<typename Job, size_t Domains>
struct TiledJobBatch {
    static constexpr uint32_t ArgumentWords{3 * Domains};

    TiledJobBatch(mtl::BufferContext &ctx, uint32_t widest_scratch_words, uint32_t most_jobs)
        : Scratch{ctx, uint64_t(widest_scratch_words + ArgumentWords) * sizeof(uint32_t), SlotType::Buffer},
          JobBuffer{ctx, uint64_t(most_jobs) * sizeof(Job), SlotType::Buffer},
          TileBuffer{ctx, uint64_t(widest_scratch_words / 32u + most_jobs * 8u) * sizeof(uvec2), SlotType::Buffer} {}

    // Starts a submit's job list over a fresh scratch layout.
    void Begin() {
        Jobs.clear();
        for (auto &tiles : Tiles) tiles.clear();
        ScratchWords = ArgumentWords;
    }
    // Claims `words` of scratch for the next job and returns their first word.
    uint32_t AllocateScratch(uint32_t words) { return std::exchange(ScratchWords, ScratchWords + words); }
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
    // The submit's scratch words, for staging inputs before the passes are encoded and reading results after they complete.
    std::span<uint32_t> ScratchSpan() const { return Scratch.GetMutableSpan<uint32_t>({0, ScratchWords}); }

    // Uploads the jobs and tiles, refills the indirect argument words, and encodes the passes in order into `encoder`.
    // `pc` carries the pass-specific push constants, whose job, tile-map, and scratch slots this fills.
    // Each pass publishes its bindless-buffer writes before the next pass reads them.
    template<typename PC>
    void Encode(const mtl::BindlessSet &slots, const MeshPipelines &pipelines, PC pc, std::span<const TiledPass> passes, MTL::ComputeCommandEncoder *encoder) {
        std::array<uint32_t, Domains> first_tile{};
        std::vector<uvec2> tiles;
        for (size_t d = 0; d < Domains; ++d) {
            first_tile[d] = uint32_t(tiles.size());
            tiles.insert(tiles.end(), Tiles[d].begin(), Tiles[d].end());
        }
        JobBuffer.Update(as_bytes(Jobs));
        TileBuffer.Update(as_bytes(tiles));
        const auto arguments = ScratchSpan();
        for (size_t d = 0; d < Domains; ++d) {
            arguments[3 * d] = uint32_t(Tiles[d].size());
            arguments[3 * d + 1] = arguments[3 * d + 2] = 1u;
        }
        pc.JobsSlot = JobBuffer.Slot;
        pc.TileMapSlot = TileBuffer.Slot;
        pc.ScratchSlot = Scratch.Slot;
        for (const auto &pass : passes) {
            const auto groups = pass.Domain == PerJob ? Jobs.size() : Tiles[pass.Domain].size();
            if (groups == 0) continue;
            encoder->setComputePipelineState(pipelines[pass.Pipeline].State());
            slots.UseResources(encoder);
            encoder->setBuffer(slots.Table(), 0, BufferIndex_Bindless);
            pc.FirstTile = pass.Domain == PerJob ? 0u : first_tile[pass.Domain];
            if constexpr (requires { pc.PassParameter; }) pc.PassParameter = pass.Parameter;
            encoder->setBytes(&pc, sizeof(pc), BufferIndex_PushConstants);
            // Eight simd-group sums and the threadgroup total, padded to Metal's 16-byte granule.
            encoder->setThreadgroupMemoryLength(48, 0);
            if (pass.Indirect) encoder->dispatchThreadgroups(*Scratch, uint64_t(3 * pass.Domain) * sizeof(uint32_t), MTL::Size(256, 1, 1));
            else encoder->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(256, 1, 1));
            encoder->memoryBarrier(MTL::BarrierScopeBuffers);
        }
    }

    // Encodes the passes into a command buffer of their own and waits for its completion.
    template<typename PC>
    void Submit(const mtl::Context &ctx, const mtl::BindlessSet &slots, const MeshPipelines &pipelines, PC pc, std::span<const TiledPass> passes) {
        auto *command_buffer = ctx.Queue->commandBuffer();
        auto *encoder = command_buffer->computeCommandEncoder();
        Encode(slots, pipelines, pc, passes, encoder);
        encoder->endEncoding();
        // Encoding grows the job and tile buffers, so residency commits after it.
        ctx.CommitResidency();
        command_buffer->commit();
        command_buffer->waitUntilCompleted();
    }

    std::vector<Job> Jobs;
    std::array<std::vector<uvec2>, Domains> Tiles;
    uint32_t ScratchWords{0};
    mtl::Buffer Scratch, JobBuffer, TileBuffer;
};
