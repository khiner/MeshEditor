#include "mesh/EditVisibility.h"
#include "Profile.h"
#include "gpu/EditVisibilityPushConstants.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/ScratchPasses.h"
#include "metal/Dispatch.h"
#include "state/Registry.h"

bool EditVisibility(state::Registry &r, std::span<const uint32_t> ids, Element element, EditVisibilityOperation operation) {
    const auto source = SelectionIndexOf(element);
    if (source >= 3u || ids.empty()) return false;
    const profile::CpuScope scope{"EditVisibility"};
    const bool reveal = operation == EditVisibilityOperation::Reveal || operation == EditVisibilityOperation::RevealSelected;
    const bool select = operation == EditVisibilityOperation::RevealSelected;
    auto &meshes = r.Resources.get<MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &pipelines = GetMeshPipelines(r);
    mtl::ComputeChain chain{meshes.BufferContext()};
    auto &scratch = chain.Scratch;
    struct Job {
        uint32_t Id;
        EditVisibilityPushConstants Pc;
        std::array<uint32_t, 3> Words;
    };
    std::vector<Job> jobs;
    for (const auto id : ids) {
        const auto &record = meshes.Get(id);
        Job job{.Id = id};
        auto &pc = job.Pc;
        pc.Connectivity = meshes.GetConnectivityRef(id);
        pc.FaceCount = a.FaceTriangles.Count(record.FaceData);
        pc.ScratchSlot = scratch.Buffer.Slot;
        pc.Source = source;
        pc.Unselected = operation == EditVisibilityOperation::HideUnselected ? 1u : 0u;
        for (uint32_t d = 0u; d < 3u; ++d) {
            pc.Blocks[d] = BlocksOf(meshes, id, SelectionDomains[d]);
            pc.HiddenSlot[d] = meshes.GetHiddenSlot(SelectionElements[d]);
            pc.SelectionSlot[d] = meshes.GetSelectionSlot(SelectionElements[d]);
            job.Words[d] = std::max(pc.Blocks[d].Count * MeshElementBlockWords, 1u);
            pc.Changes[d] = scratch.Allocate(job.Words[d]).Offset;
            FillScratch(chain, pipelines, pc.Changes[d], job.Words[d], 0u);
        }
        const auto threads = [&](uint32_t d) { return pc.Blocks[d].Count * MeshElementBlockSize; };
        if (reveal) {
            for (uint32_t d = 0u; d < 3u; ++d) {
                auto reveal_pc = pc;
                reveal_pc.Source = d;
                chain.Threads(pipelines[MeshPass::EditVisibilityReveal], reveal_pc, threads(d));
            }
        } else {
            chain.Threads(pipelines[MeshPass::EditVisibilitySource], pc, threads(source));
            if (source == 0u) {
                chain.Threads(pipelines[MeshPass::EditVisibilityEdges], pc, threads(1u));
                chain.Threads(pipelines[MeshPass::EditVisibilityFaces], pc, threads(2u));
            } else if (source == 1u) {
                chain.Threads(pipelines[MeshPass::EditVisibilityFaces], pc, threads(2u));
                chain.Threads(pipelines[MeshPass::EditVisibilityVertices], pc, threads(0u));
            } else {
                chain.Threads(pipelines[MeshPass::EditVisibilityEdges], pc, threads(1u));
                chain.Threads(pipelines[MeshPass::EditVisibilityVertices], pc, threads(0u));
            }
        }
        jobs.push_back(job);
    }
    chain.Submit();
    std::vector<MeshStore::SelectionUpdate> updates;
    uint64_t changed = 0u;
    for (const auto &job : jobs) {
        const auto id = job.Id;
        // Change words by handle map onto mask words by arena address through the record's blocks.
        std::array<std::vector<std::pair<uint32_t, uint32_t>>, 3> words;
        for (uint32_t d = 0u; d < 3u; ++d) {
            const auto blocks = meshes.GetBlockList(id, SelectionDomains[d]).Blocks;
            const auto membership = WithDomain(a, SelectionDomains[d], [](const auto &arena) { return arena.Blocks.Buffer.template GetSpan<MeshElementBlock>(); });
            const auto changes = scratch.Get({job.Pc.Changes[d], job.Words[d]});
            for (const auto block : blocks) {
                for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) {
                    if (const auto bits = changes[membership[block].Local * MeshElementBlockWords + w]) words[d].emplace_back(block * MeshElementBlockWords + w, bits);
                }
            }
            std::ranges::sort(words[d], {}, &std::pair<uint32_t, uint32_t>::first);
        }
        if (std::ranges::all_of(words, [](const auto &domain) { return domain.empty(); })) continue;
        auto &update = updates.emplace_back(MeshStore::SelectionUpdate{.StoreId = id, .Source = element});
        for (uint32_t d = 0u; d < 3u; ++d) {
            std::vector<uint32_t> blocks;
            for (const auto [word, bits] : words[d]) {
                changed += uint32_t(std::popcount(bits));
                if (blocks.empty() || blocks.back() != word / MeshElementBlockWords) blocks.push_back(word / MeshElementBlockWords);
                update.Seeds.push_back({d, word, bits});
            }
            const auto write = [&](bool set) {
                return [&, set, at = words[d].begin()](uint32_t block, auto &out) mutable {
                    while (at != words[d].end() && at->first / MeshElementBlockWords == block) {
                        const auto [word, bits] = *at++;
                        if (set) out[word % MeshElementBlockWords] |= bits;
                        else out[word % MeshElementBlockWords] &= ~bits;
                    }
                };
            };
            meshes.EditHiddenBlocks(SelectionElements[d], blocks, write(!reveal));
            if (!reveal || (select && d == source)) meshes.EditSelectionBlocks(SelectionElements[d], blocks, write(reveal));
        }
        auto &summary = meshes.WriteSelectionSummary(id);
        summary.ActiveHandle = InvalidOffset;
    }
    if (updates.empty()) return false;
    profile::RecordCounter("VisibilityChangedElements", changed);
    meshes.UpdateSelection(r, chain, updates);
    chain.Submit();
    for (const auto &update : updates) meshes.PublishSelectionSummary(update.StoreId);
    return true;
}
