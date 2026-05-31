#pragma once

#include <entt/entity/fwd.hpp>

#include <filesystem>
#include <memory>

struct SvgResource;

// LoadSvg accumulates uploads into a batch, and SubmitSvgUpload flushes them in one submit.
struct SvgUploadBatch {
    struct Impl;
    explicit SvgUploadBatch(entt::registry &);
    ~SvgUploadBatch();
    SvgUploadBatch(const SvgUploadBatch &) = delete;
    SvgUploadBatch &operator=(const SvgUploadBatch &) = delete;
    std::unique_ptr<Impl> Imp;
};

std::unique_ptr<SvgResource> LoadSvg(SvgUploadBatch &, std::filesystem::path);
void SubmitSvgUpload(SvgUploadBatch &);

// Convenience method to load and submit a one-shot batch.
std::unique_ptr<SvgResource> LoadSvg(entt::registry &, std::filesystem::path);
