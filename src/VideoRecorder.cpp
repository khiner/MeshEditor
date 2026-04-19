#include "VideoRecorder.h"

#include "vulkan/Image.h" // mvk::FindMemoryType

#include <cstdlib>
#include <format>
#include <print>

void VideoRecorder::PipeCloser::operator()(std::FILE *p) const noexcept {
    if (p) ::pclose(p);
}

namespace {
std::string BuildFfmpegCommand(const std::filesystem::path &out, vk::Extent2D extent, int fps) {
    // `-y` overwrite, `-loglevel warning` mutes per-frame progress.
    // Input: raw BGRA frames on stdin with declared size/framerate.
    // `-vf vflip`: the viewport uses a negative-height Vulkan viewport so row 0 in image
    // memory is the bottom of the screen; flip rows to get upright video.
    // Output: H.264 in yuv420p
    return std::format(
        "ffmpeg -y -loglevel warning -f rawvideo -pix_fmt bgra -s {}x{} -r {} -i - "
        "-vf vflip -c:v libx264 -pix_fmt yuv420p -preset medium -crf 18 \"{}\"",
        extent.width, extent.height, fps, out.string()
    );
}

bool IsFfmpegAvailable() {
    return std::system("command -v ffmpeg >/dev/null 2>&1") == 0;
}
} // namespace

VideoRecorder::VideoRecorder(
    const SceneVulkanResources &vk_res,
    std::filesystem::path output_path, vk::Offset3D offset, vk::Extent2D extent, int fps
) : Device{vk_res.Device}, Queue{vk_res.Queue}, Offset{offset}, Ex{extent},
    FrameBytes{extent.width * extent.height * 4} {
    if (extent.width == 0 || extent.height == 0) {
        std::println(stderr, "VideoRecorder: viewport extent is zero; not recording.");
        return;
    }
    if (!IsFfmpegAvailable()) {
        std::println(stderr, "VideoRecorder: 'ffmpeg' not found on PATH; not recording.");
        return;
    }

    CommandPool = Device.createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, vk_res.QueueFamily});
    CommandBuffer = std::move(Device.allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, 1}).front());
    Fence = Device.createFenceUnique({});

    Buffer = Device.createBufferUnique({{}, FrameBytes, vk::BufferUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive});
    const auto mem_reqs = Device.getBufferMemoryRequirements(*Buffer);
    Memory = Device.allocateMemoryUnique({mem_reqs.size, mvk::FindMemoryType(vk_res.PhysicalDevice, mem_reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)});
    Device.bindBufferMemory(*Buffer, *Memory, 0);
    Mapped = Device.mapMemory(*Memory, 0, FrameBytes);

    const auto cmd = BuildFfmpegCommand(output_path, extent, fps);
    std::println("VideoRecorder: {}x{} @ {}fps -> {}", extent.width, extent.height, fps, output_path.string());
    Pipe.reset(::popen(cmd.c_str(), "w"));
    if (!Pipe) std::println(stderr, "VideoRecorder: popen failed");
}

VideoRecorder::~VideoRecorder() { Stop(); }

void VideoRecorder::Stop() {
    if (Pipe) {
        std::fflush(Pipe.get());
        const int status = ::pclose(Pipe.release());
        std::println("VideoRecorder: wrote {} frames (ffmpeg status={})", FrameCount, status);
    }
    if (Mapped) {
        Device.unmapMemory(*Memory);
        Mapped = nullptr;
    }
}

void VideoRecorder::CaptureFrame(vk::Image image) {
    if (!Pipe) return;

    auto cb = *CommandBuffer;
    cb.reset({});
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    const vk::ImageSubresourceRange color_range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eTransfer,
        {}, {}, {},
        vk::ImageMemoryBarrier{
            vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eTransferRead, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferSrcOptimal, {}, {}, image, color_range
        }
    );
    cb.copyImageToBuffer(
        image, vk::ImageLayout::eTransferSrcOptimal, *Buffer,
        vk::BufferImageCopy{0, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, Offset, {Ex.width, Ex.height, 1}}
    );
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
        {}, {}, {},
        vk::ImageMemoryBarrier{
            vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, {}, {}, image, color_range
        }
    );
    cb.end();

    Queue.submit(vk::SubmitInfo{{}, {}, cb, {}}, *Fence);
    if (Device.waitForFences(*Fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess) {
        std::println(stderr, "VideoRecorder: fence wait failed; stopping.");
        Stop();
        return;
    }
    Device.resetFences(*Fence);

    if (const auto written = std::fwrite(Mapped, 1, FrameBytes, Pipe.get()); written != FrameBytes) {
        std::println(stderr, "VideoRecorder: pipe write short ({}/{}); stopping.", written, FrameBytes);
        Stop();
        return;
    }
    ++FrameCount;
}
