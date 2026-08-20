#include "AudioDevice.h"

#include "CoreAudioTypes.h"
#include "AudioSystem.h"
#include "action/Audio.h" // Replace<AudioOutputConfig>
#include "action/Emit.h"
#include "ui/FieldEdit.h"

#include "imgui.h"

#include <entt/entity/registry.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <ranges>
#include <stdexcept>

#include <AudioUnit/AudioUnit.h>
#include <CoreAudio/CoreAudio.h>
#include <os/workgroup.h>

template<> struct FieldLimits<&AudioOutputMix::Volume> : Within<0., 1.> {};
static_assert(std::atomic<float>::is_always_lock_free, "The render callback requires lock-free gain updates.");

namespace {
constexpr AudioObjectPropertyAddress DeviceNameAddress{kAudioObjectPropertyName, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
constexpr AudioObjectPropertyAddress DevicesAddress{kAudioHardwarePropertyDevices, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
constexpr AudioObjectPropertyAddress DefaultOutputAddress{kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
constexpr AudioObjectPropertyAddress OutputStreamsAddress{kAudioDevicePropertyStreamConfiguration, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain};
constexpr AudioObjectPropertyAddress SampleRateAddress{kAudioDevicePropertyNominalSampleRate, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain};
constexpr AudioObjectPropertyAddress SampleRatesAddress{kAudioDevicePropertyAvailableNominalSampleRates, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain};

void Check(OSStatus status, std::string_view operation) {
    if (status != noErr) throw std::runtime_error(std::format("CoreAudio {} failed (OSStatus {})", operation, status));
}

template<typename T>
T GetProperty(AudioObjectID object, const AudioObjectPropertyAddress &address, std::string_view operation) {
    T value{};
    UInt32 size = sizeof(value);
    Check(AudioObjectGetPropertyData(object, &address, 0, nullptr, &size, &value), operation);
    return value;
}

template<typename T>
std::vector<T> GetPropertyVector(AudioObjectID object, const AudioObjectPropertyAddress &address, std::string_view operation) {
    UInt32 size = 0;
    Check(AudioObjectGetPropertyDataSize(object, &address, 0, nullptr, &size), operation);
    std::vector<T> values(size / sizeof(T));
    if (size) Check(AudioObjectGetPropertyData(object, &address, 0, nullptr, &size, values.data()), operation);
    values.resize(size / sizeof(T));
    return values;
}

std::string DeviceName(AudioDeviceID device) {
    CFStringRef name = GetProperty<CFStringRef>(device, DeviceNameAddress, "reading the device name");
    const auto capacity = CFStringGetMaximumSizeForEncoding(CFStringGetLength(name), kCFStringEncodingUTF8) + 1;
    std::string result(size_t(capacity), '\0');
    if (!CFStringGetCString(name, result.data(), capacity, kCFStringEncodingUTF8)) result.clear();
    else result.resize(result.find('\0'));
    CFRelease(name);
    return result;
}

bool HasOutput(AudioDeviceID device) {
    UInt32 size = 0;
    if (AudioObjectGetPropertyDataSize(device, &OutputStreamsAddress, 0, nullptr, &size) != noErr || size < sizeof(AudioBufferList)) return false;
    std::vector<std::byte> storage(size);
    auto *streams = reinterpret_cast<AudioBufferList *>(storage.data());
    if (AudioObjectGetPropertyData(device, &OutputStreamsAddress, 0, nullptr, &size, streams) != noErr) return false;
    for (UInt32 i = 0; i < streams->mNumberBuffers; ++i) {
        if (streams->mBuffers[i].mNumberChannels) return true;
    }
    return false;
}

AudioDeviceID RefreshDevices(std::vector<std::string> &names, std::string_view requested_name) {
    names.clear();
    AudioDeviceID selected = kAudioObjectUnknown;
    for (AudioDeviceID device : GetPropertyVector<AudioDeviceID>(kAudioObjectSystemObject, DevicesAddress, "enumerating devices")) {
        if (!HasOutput(device)) continue;
        auto name = DeviceName(device);
        if (name == requested_name) selected = device;
        names.emplace_back(std::move(name));
    }
    return selected != kAudioObjectUnknown ? selected : GetProperty<AudioDeviceID>(kAudioObjectSystemObject, DefaultOutputAddress, "reading the default output device");
}

std::vector<uint32_t> SampleRates(AudioDeviceID device, uint32_t current_rate) {
    const auto ranges = GetPropertyVector<AudioValueRange>(device, SampleRatesAddress, "reading device sample rates");
    // Discrete devices publish one zero-width range per rate. For devices publishing a continuous
    // range, expose the conventional audio rates that the HAL says are valid, plus its endpoints.
    constexpr std::array<uint32_t, 16> ConventionalRates{
        8'000, 11'025, 16'000, 22'050, 24'000, 32'000, 44'100, 48'000,
        64'000, 88'200, 96'000, 128'000, 176'400, 192'000, 352'800, 384'000
    };
    std::vector<uint32_t> result;
    const auto add = [&](double rate) {
        if (rate > 0 && rate <= double(UINT32_MAX)) result.emplace_back(uint32_t(std::lround(rate)));
    };
    for (const auto &range : ranges) {
        add(range.mMinimum);
        add(range.mMaximum);
        for (uint32_t rate : ConventionalRates) {
            if (rate >= range.mMinimum && rate <= range.mMaximum) result.emplace_back(rate);
        }
    }
    if (current_rate) result.emplace_back(current_rate);
    std::ranges::sort(result);
    auto [first_duplicate, end] = std::ranges::unique(result);
    result.erase(first_duplicate, end);
    return result;
}

// The scheduling group the device's IO thread belongs to, which threads rendering alongside it join.
// Null when the device publishes none.
void *DeviceWorkgroup(uint32_t device_object_id) {
    os_workgroup_t workgroup{nullptr};
    UInt32 size = sizeof(workgroup);
    const AudioObjectPropertyAddress address{kAudioDevicePropertyIOThreadOSWorkgroup, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
    if (AudioObjectGetPropertyData(device_object_id, &address, 0, nullptr, &size, static_cast<void *>(&workgroup)) != noErr) return nullptr;
    return static_cast<void *>(workgroup);
}

// Give back the reference DeviceWorkgroup handed out. Other holders keep their own.
void ReleaseWorkgroup(void *workgroup) {
    if (workgroup) os_release(static_cast<os_workgroup_t>(workgroup));
}

OSStatus DataCallback(
    void *context, AudioUnitRenderActionFlags *flags, const AudioTimeStamp *, UInt32, UInt32 frame_count, AudioBufferList *output
) noexcept {
    auto &res = *static_cast<AudioDeviceResource *>(context);
    if (!frame_count || !output || output->mNumberBuffers == 0 || !output->mBuffers[0].mData) return noErr;

    auto *frames = static_cast<float *>(output->mBuffers[0].mData);
    try {
        ProcessAudio(*res.R, res.Viewport, frames, frame_count, true);

        const float from = res.RenderGain;
        const float to = res.TargetGain.load(std::memory_order_relaxed);
        if (from != to) {
            const float step = (to - from) / float(frame_count);
            float gain = from;
            for (UInt32 i = 0; i < frame_count; ++i) frames[i] *= gain += step;
            res.RenderGain = to;
        } else if (to == 0.f) {
            std::fill_n(frames, frame_count, 0.f);
        } else if (to != 1.f) {
            for (UInt32 i = 0; i < frame_count; ++i) frames[i] *= to;
        }
    } catch (...) {
        std::fill_n(frames, frame_count, 0.f);
        if (flags) *flags |= kAudioUnitRenderAction_OutputIsSilence;
    }
    return noErr;
}

void CloseDevice(AudioDeviceResource &res) {
    if (res.OutputUnit) {
        auto unit = static_cast<AudioUnit>(res.OutputUnit);
        if (res.Started) AudioOutputUnitStop(unit);
        AudioUnitUninitialize(unit);
        AudioComponentInstanceDispose(unit);
    }
    res.OutputUnit = nullptr;
    res.SampleRate = 0;
    res.Initialized = false;
    res.Started = false;
    ReleaseWorkgroup(res.RenderWorkgroup);
    res.RenderWorkgroup = nullptr;
}

AudioUnit OpenDevice(AudioDeviceResource &res, AudioDeviceID device, uint32_t sample_rate) {
    const AudioComponentDescription description{kAudioUnitType_Output, kAudioUnitSubType_HALOutput, kAudioUnitManufacturer_Apple, 0, 0};
    const AudioComponent component = AudioComponentFindNext(nullptr, &description);
    if (!component) throw std::runtime_error("CoreAudio HAL output component is unavailable");

    AudioUnit unit{nullptr};
    Check(AudioComponentInstanceNew(component, &unit), "creating the HAL output unit");
    try {
        const UInt32 enabled = 1;
        Check(AudioUnitSetProperty(unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, &enabled, sizeof(enabled)), "enabling output");
        const UInt32 disabled = 0;
        Check(AudioUnitSetProperty(unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, &disabled, sizeof(disabled)), "disabling input");
        Check(AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &device, sizeof(device)), "selecting the output device");

        const auto format = MonoFloatAudioFormat(sample_rate);
        Check(AudioUnitSetProperty(unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &format, sizeof(format)), "setting the mono float stream format");

        const AURenderCallbackStruct callback{.inputProc = DataCallback, .inputProcRefCon = &res};
        Check(AudioUnitSetProperty(unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Global, 0, &callback, sizeof(callback)), "installing the render callback");
        Check(AudioUnitInitialize(unit), "initializing the HAL output unit");
        return unit;
    } catch (...) {
        AudioComponentInstanceDispose(unit);
        throw;
    }
}

std::string SampleRateName(const AudioDeviceResource &res, uint32_t sample_rate) {
    const auto &rates = res.NativeSampleRates;
    const bool is_native = std::find(rates.begin(), rates.end(), sample_rate) != rates.end();
    return std::format("{}{}", sample_rate, is_native ? "*" : "");
}
} // namespace

AudioDeviceResource::AudioDeviceResource(entt::registry &r, entt::entity viewport) : R(&r), Viewport(viewport) {}
AudioDeviceResource::~AudioDeviceResource() {
    CloseDevice(*this);
}

// Reopen only when the device or rate changes. Volume, mute, and on/off apply live.
void ReconcileAudioDevice(AudioDeviceResource &res, const AudioOutputConfig &config, const AudioOutputMix &mix) {
    if (!res.Initialized || config.DeviceName != res.DeviceName || config.SampleRate != res.RequestedSampleRate) {
        CloseDevice(res);

        const AudioDeviceID device = RefreshDevices(res.OutDeviceNames, config.DeviceName);
        if (config.SampleRate) {
            const Float64 requested_rate = config.SampleRate;
            Check(AudioObjectSetPropertyData(device, &SampleRateAddress, 0, nullptr, sizeof(requested_rate), &requested_rate), "setting the device sample rate");
        }
        const auto sample_rate = uint32_t(std::lround(GetProperty<Float64>(device, SampleRateAddress, "reading the device sample rate")));
        res.NativeSampleRates = SampleRates(device, sample_rate);
        res.OutputUnit = OpenDevice(res, device, sample_rate);
        res.SampleRate = sample_rate;
        res.RenderWorkgroup = DeviceWorkgroup(device);
        res.Initialized = true;
        res.DeviceName = config.DeviceName;
        res.RequestedSampleRate = config.SampleRate;
    }

    ApplyAudioMix(res, mix);
}

void ApplyAudioMix(AudioDeviceResource &res, const AudioOutputMix &mix) {
    if (!res.Initialized) return;
    const float target_gain = mix.Muted ? 0.f : mix.Volume;
    res.TargetGain.store(target_gain, std::memory_order_relaxed);
    auto unit = static_cast<AudioUnit>(res.OutputUnit);
    if (mix.On && !res.Started) {
        res.RenderGain = target_gain;
        Check(AudioOutputUnitStart(unit), "starting the output device");
        res.Started = true;
    } else if (!mix.On && res.Started) {
        Check(AudioOutputUnitStop(unit), "stopping the output device");
        res.Started = false;
        res.RenderGain = target_gain;
    } else if (!res.Started) {
        res.RenderGain = target_gain;
    }
}

void DrawAudioDeviceControls(entt::registry &r, entt::entity viewport) {
    using namespace ImGui;
    const auto &config = r.get<const AudioOutputConfig>(viewport);
    const auto &mix = r.get<const AudioOutputMix>(viewport);
    const auto &res = r.ctx().get<const AudioDeviceResource>();
    ui::Edit f{r, viewport};

    f.Check<&AudioOutputMix::On>("On");
    if (!mix.On) {
        TextUnformatted("Audio device: Not started");
        return;
    }

    if (BeginCombo("Output device", config.DeviceName.empty() ? "System default" : config.DeviceName.c_str())) {
        for (const auto &name : res.OutDeviceNames) {
            const bool is_selected = name == config.DeviceName;
            if (Selectable(name.c_str(), is_selected) && !is_selected) action::Emit(action::Replace<AudioOutputConfig>{.Entity = viewport, .Value = {.DeviceName = name, .SampleRate = 0}});
            if (is_selected) SetItemDefaultFocus();
        }
        EndCombo();
    }
    if (BeginCombo("Sample rate", SampleRateName(res, res.SampleRate).c_str())) {
        for (const uint32_t option : res.NativeSampleRates) {
            const bool is_selected = option == res.SampleRate;
            if (Selectable(SampleRateName(res, option).c_str(), is_selected) && !is_selected) f.Set<&AudioOutputConfig::SampleRate>(option);
            if (is_selected) SetItemDefaultFocus();
        }
        EndCombo();
    }

    f.Check<&AudioOutputMix::Muted>("Muted");
    SameLine();
    if (mix.Muted) BeginDisabled();
    f.Slider<&AudioOutputMix::Volume>("Volume");
    if (mix.Muted) EndDisabled();
}
