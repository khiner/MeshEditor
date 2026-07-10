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

std::unique_ptr<SvgResource> LoadSvg(SvgUploadBatch &, const std::filesystem::path &);
void SubmitSvgUpload(SvgUploadBatch &);
