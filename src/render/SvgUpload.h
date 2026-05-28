#pragma once

#include <entt/entity/fwd.hpp>

#include <filesystem>
#include <memory>

struct SvgResource;

// Vulkan-free SVG upload state. LoadSvg records uploads into it; SubmitSvgUpload flushes them
// in one GPU submit. Opaque (pimpl) so icon/UI loaders need not pull the upload (vulkan) headers.
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

// Load and submit a one-shot batch.
std::unique_ptr<SvgResource> LoadSvg(entt::registry &, std::filesystem::path);
