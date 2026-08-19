#include "FileDialog.h"

#import <AppKit/AppKit.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#include <utility>

namespace FileDialog {
namespace {
NSArray<UTType *> *Types(const char *extensions) {
    auto *const types = [NSMutableArray array];
    for (NSString *extension in [[NSString stringWithUTF8String:extensions] componentsSeparatedByString:@";"]) {
        if (auto *const type = [UTType typeWithFilenameExtension:extension]) [types addObject:type];
    }
    return types;
}

void Show(NSSavePanel *panel, OnPick callback) {
    [panel beginWithCompletionHandler:[callback = std::move(callback), panel](NSModalResponse response) {
        if (response == NSModalResponseOK && panel.URL) callback(std::filesystem::path{panel.URL.fileSystemRepresentation});
    }];
}
} // namespace

void ShowOpen(const char *extensions, OnPick callback) {
    auto *const panel = [NSOpenPanel openPanel];
    panel.allowedContentTypes = Types(extensions);
    Show(panel, std::move(callback));
}

void ShowSave(const char *extensions, const char *default_name, OnPick callback) {
    auto *const panel = [NSSavePanel savePanel];
    panel.allowedContentTypes = Types(extensions);
    panel.nameFieldStringValue = [NSString stringWithUTF8String:default_name];
    panel.extensionHidden = NO;
    Show(panel, std::move(callback));
}

void ShowPickFolder(OnPick callback) {
    auto *const panel = [NSOpenPanel openPanel];
    panel.canChooseFiles = NO;
    panel.canChooseDirectories = YES;
    Show(panel, std::move(callback));
}
} // namespace FileDialog
