#include "AudioDevice.h"

#include "AudioOutputConfig.h"
#include "AudioSystem.h"
#include "CoreAudioTypes.h"

#include <algorithm>
#include <cmath>
#include <format>

static_assert(std::atomic<float>::is_always_lock_free, "The render callback requires lock-free gain updates.");

namespace {
constexpr AudioObjectPropertyAddress DeviceNameAddress{kAudioObjectPropertyName, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
constexpr AudioObjectPropertyAddress DevicesAddress{kAudioHardwarePropertyDevices, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
constexpr AudioObjectPropertyAddress DefaultOutputAddress{kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
constexpr AudioObjectPropertyAddress OutputStreamsAddress{kAudioDevicePropertyStreamConfiguration, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain};
// The nominal rate is a global property, and the device notifies its listeners only under this scope.
constexpr AudioObjectPropertyAddress SampleRateAddress{kAudioDevicePropertyNominalSampleRate, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
constexpr AudioObjectPropertyAddress SampleRatesAddress{kAudioDevicePropertyAvailableNominalSampleRates, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain};

void Check(OSStatus status, std::string_view operation) {
    if (status != noErr) throw std::runtime_error(std::format("CoreAudio {} failed (OSStatus {})", operation, status));
}

template<typename T>
T GetProperty(AudioObjectID object, const AudioObjectPropertyAddress &address, std::string_view operation) {
    T value{};
    UInt32 size = sizeof(value);
    Check(AudioObjectGetPropertyData(object, &address, 0, nullptr, &size, static_cast<void *>(&value)), operation);
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
    const auto *name = GetProperty<CFStringRef>(device, DeviceNameAddress, "reading the device name");
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
    for (const AudioDeviceID device : GetPropertyVector<AudioDeviceID>(kAudioObjectSystemObject, DevicesAddress, "enumerating devices")) {
        if (!HasOutput(device)) continue;
        auto name = DeviceName(device);
        if (name == requested_name) selected = device;
        names.emplace_back(std::move(name));
    }
    return selected != kAudioObjectUnknown ? selected : GetProperty<AudioDeviceID>(kAudioObjectSystemObject, DefaultOutputAddress, "reading the default output device");
}

std::vector<uint32_t> SampleRates(AudioDeviceID device, uint32_t current_rate) {
    const auto ranges = GetPropertyVector<AudioValueRange>(device, SampleRatesAddress, "reading device sample rates");
    // Discrete devices publish one zero-width range per rate.
    // For continuous ranges, expose valid conventional rates and the endpoints.
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

// Release the caller's DeviceWorkgroup reference.
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
        ProcessAudio(*res.R, frames, frame_count, true);

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

// The device reports its nominal rate from a CoreAudio thread, and the main thread follows it.
OSStatus ReportSampleRate(AudioObjectID device, UInt32, const AudioObjectPropertyAddress *, void *context) noexcept {
    Float64 rate{};
    UInt32 size = sizeof(rate);
    if (AudioObjectGetPropertyData(device, &SampleRateAddress, 0, nullptr, &size, &rate) == noErr) {
        static_cast<AudioDeviceResource *>(context)->ReportedSampleRate.store(uint32_t(std::lround(rate)), std::memory_order_release);
    }
    return noErr;
}

void CloseUnit(AudioDeviceResource &res) {
    if (res.OutputUnit) {
        auto *unit = static_cast<AudioUnit>(res.OutputUnit);
        if (res.Started) AudioOutputUnitStop(unit);
        AudioUnitUninitialize(unit);
        AudioComponentInstanceDispose(unit);
    }
    res.OutputUnit = nullptr;
    res.Started = false;
}

void CloseDevice(AudioDeviceResource &res) {
    CloseUnit(res);
    if (res.Device) AudioObjectRemovePropertyListener(res.Device, &SampleRateAddress, &ReportSampleRate, &res);
    res.Device = kAudioObjectUnknown;
    ReleaseWorkgroup(res.RenderWorkgroup);
    res.RenderWorkgroup = nullptr;
}

// The unit renders at the device's rate, and its stream format is fixed while it is initialized.
void OpenUnit(AudioDeviceResource &res, AudioDeviceInfo &info, uint32_t sample_rate) {
    const AudioDeviceID device = res.Device;
    const AudioComponentDescription description{kAudioUnitType_Output, kAudioUnitSubType_HALOutput, kAudioUnitManufacturer_Apple, 0, 0};
    auto *const component = AudioComponentFindNext(nullptr, &description);
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
    } catch (...) {
        AudioComponentInstanceDispose(unit);
        throw;
    }
    res.OutputUnit = unit;
    info.SampleRate = sample_rate;
    info.NativeSampleRates = SampleRates(device, sample_rate);
}

} // namespace

AudioDeviceResource::AudioDeviceResource(state::Registry &r) : R(&r) {}
AudioDeviceResource::~AudioDeviceResource() {
    CloseDevice(*this);
}

bool ReconcileAudioDevice(AudioDeviceResource &res, AudioDeviceInfo &info, const AudioOutputConfig &config) {
    bool reopened = false;
    if (res.Device == kAudioObjectUnknown || config.DeviceName != res.DeviceName) {
        CloseDevice(res);
        res.Device = RefreshDevices(info.OutDeviceNames, config.DeviceName);
        res.DeviceName = config.DeviceName;
        res.RequestedSampleRate = 0;
        const auto sample_rate = uint32_t(std::lround(GetProperty<Float64>(res.Device, SampleRateAddress, "reading the device sample rate")));
        res.ReportedSampleRate.store(sample_rate, std::memory_order_release);
        Check(AudioObjectAddPropertyListener(res.Device, &SampleRateAddress, &ReportSampleRate, &res), "listening for the device sample rate");
        res.RenderWorkgroup = DeviceWorkgroup(res.Device);
        OpenUnit(res, info, sample_rate);
        reopened = true;
    }
    if (config.SampleRate != res.RequestedSampleRate) {
        res.RequestedSampleRate = config.SampleRate;
        if (config.SampleRate) {
            const Float64 requested_rate = config.SampleRate;
            Check(AudioObjectSetPropertyData(res.Device, &SampleRateAddress, 0, nullptr, sizeof(requested_rate), &requested_rate), "requesting the device sample rate");
        }
    }
    if (const auto reported = res.ReportedSampleRate.load(std::memory_order_acquire); reported != info.SampleRate) {
        CloseUnit(res);
        OpenUnit(res, info, reported);
        reopened = true;
    }
    return reopened;
}

void ApplyAudioMix(AudioDeviceResource &res, const AudioOutputMix &mix) {
    if (!res.OutputUnit) return;
    const float target_gain = mix.Muted ? 0.f : mix.Volume;
    res.TargetGain.store(target_gain, std::memory_order_relaxed);
    auto *unit = static_cast<AudioUnit>(res.OutputUnit);
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
