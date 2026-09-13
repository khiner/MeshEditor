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

void ShowOpen(const char *extensions, OnPick callback, bool directories) {
    auto *const panel = [NSOpenPanel openPanel];
    panel.allowedContentTypes = Types(extensions);
    panel.canChooseDirectories = directories;
    Show(panel, std::move(callback));
}

void ShowSave(const char *extensions, const std::filesystem::path &default_path, OnPick callback) {
    auto *const panel = [NSSavePanel savePanel];
    if (extensions) panel.allowedContentTypes = Types(extensions);
    if (default_path.has_parent_path()) panel.directoryURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:default_path.parent_path().c_str()]];
    panel.nameFieldStringValue = [NSString stringWithUTF8String:default_path.filename().c_str()];
    panel.canCreateDirectories = YES;
    panel.extensionHidden = NO;
    Show(panel, std::move(callback));
}

void ShowPickFolder(OnPick callback) {
    auto *const panel = [NSOpenPanel openPanel];
    panel.canCreateDirectories = YES;
    panel.canChooseFiles = NO;
    panel.canChooseDirectories = YES;
    Show(panel, std::move(callback));
}
} // namespace FileDialog
