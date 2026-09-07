#include "MacPlatform.h"

#include "Paths.h"
#include "metal/MetalCpp.h"

#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

#include <stdexcept>
#include <string>
#include <utility>

@class MeshEditorView;

namespace MacPlatform {
struct Window::Impl {
    NSWindow *NativeWindow{nil};
    MeshEditorView *View{nil};
    Events PendingEvents;
};
} // namespace MacPlatform

using MacPlatform::Window;

@interface MeshEditorView : NSView <NSApplicationDelegate, NSDraggingDestination, NSWindowDelegate>
@property(nonatomic, assign) Window::Impl *owner;
@end

@implementation MeshEditorView
- (BOOL)isFlipped {
    return YES;
}

- (void)scrollWheel:(NSEvent *)event {
    if (event.phase == NSEventPhaseCancelled) return;
    const double scale = event.hasPreciseScrollingDeltas ? 0.1 : 1.0;
    self.owner->PendingEvents.ScrollX += float(event.scrollingDeltaX * scale);
    self.owner->PendingEvents.ScrollY += float(event.scrollingDeltaY * scale);
}

- (NSDragOperation)draggingEntered:(id<NSDraggingInfo>)sender {
    return [sender.draggingPasteboard canReadObjectForClasses:@[ NSURL.class ] options:@{NSPasteboardURLReadingFileURLsOnlyKey : @YES}] ? NSDragOperationCopy : NSDragOperationNone;
}

- (BOOL)performDragOperation:(id<NSDraggingInfo>)sender {
    const auto options = @{NSPasteboardURLReadingFileURLsOnlyKey : @YES};
    const auto urls = [sender.draggingPasteboard readObjectsForClasses:@[ NSURL.class ] options:options];
    for (NSURL *url in urls) self.owner->PendingEvents.DroppedFiles.emplace_back(url.fileSystemRepresentation);
    return urls.count != 0;
}

- (BOOL)windowShouldClose:(NSWindow *)sender {
    self.owner->PendingEvents.Quit = true;
    return NO;
}
- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender {
    self.owner->PendingEvents.Quit = true;
    return NSTerminateCancel;
}
@end

static void InstallMenu(NSApplication *app) {
    auto *const main = [[NSMenu alloc] init];
    auto *const app_item = [[NSMenuItem alloc] init];
    [main addItem:app_item];
    auto *const app_menu = [[NSMenu alloc] initWithTitle:@"MeshEditor"];
    [app_menu addItem:[[NSMenuItem alloc] initWithTitle:@"Quit MeshEditor" action:@selector(terminate:) keyEquivalent:@"q"]];
    app_item.submenu = app_menu;
    app.mainMenu = main;
}

namespace MacPlatform {
void InitPaths() {
    auto *const executable = NSBundle.mainBundle.executableURL;
    if (!executable) throw std::runtime_error("Could not resolve the executable path.");

    auto *const files = NSFileManager.defaultManager;
    NSError *error = nil;
    auto *const support = [files URLForDirectory:NSApplicationSupportDirectory
                                        inDomain:NSUserDomainMask
                               appropriateForURL:nil
                                          create:YES
                                           error:&error];
    auto *const user_data = [support URLByAppendingPathComponent:@"MeshEditor" isDirectory:YES];
    if (!support || ![files createDirectoryAtURL:user_data withIntermediateDirectories:YES attributes:nil error:&error]) {
        const char *message = error.localizedDescription.UTF8String ?: "unknown error";
        throw std::runtime_error(std::string{"Could not create the user data directory: "} + message);
    }
    Paths::Init(
        executable.URLByDeletingLastPathComponent.fileSystemRepresentation,
        user_data.fileSystemRepresentation
    );
}

Window::Window() : Data{std::make_unique<Impl>()} {
    auto *const screen = NSScreen.mainScreen;
    if (!screen) throw std::runtime_error("No display is available for the MeshEditor window.");

    auto *const app = NSApplication.sharedApplication;
    [app setActivationPolicy:NSApplicationActivationPolicyRegular];
    InstallMenu(app);

    const auto style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
        NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;
    Data->NativeWindow = [[NSWindow alloc] initWithContentRect:NSZeroRect
                                                     styleMask:style
                                                       backing:NSBackingStoreBuffered
                                                         defer:NO];
    Data->NativeWindow.releasedWhenClosed = NO; // ARC owns the window through Impl.
    Data->NativeWindow.title = @"MeshEditor";
    Data->NativeWindow.acceptsMouseMovedEvents = YES;

    Data->View = [[MeshEditorView alloc] initWithFrame:NSZeroRect];
    Data->View.owner = Data.get();
    [Data->View registerForDraggedTypes:@[ NSPasteboardTypeFileURL ]];
    Data->View.wantsLayer = YES;
    Data->View.layer = [CAMetalLayer layer];
    Data->NativeWindow.contentView = Data->View;
    Data->NativeWindow.delegate = Data->View;
    app.delegate = Data->View;
    [Data->NativeWindow setFrame:screen.visibleFrame display:NO];
    [app finishLaunching];
    Data->NativeWindow.alphaValue = 0;
    [Data->NativeWindow makeKeyAndOrderFront:nil];
    [app activateIgnoringOtherApps:YES];
}

Window::~Window() {
    [Data->NativeWindow close];
}

CA::MetalLayer *Window::Layer() const { return (__bridge CA::MetalLayer *)Data->View.layer; }

void Window::Show() {
    Data->NativeWindow.alphaValue = 1;
}

Events Window::PollEvents() {
    @autoreleasepool {
        auto *const app = NSApplication.sharedApplication;
        while (auto *const event = [app nextEventMatchingMask:NSEventMaskAny
                                                    untilDate:NSDate.distantPast
                                                       inMode:NSDefaultRunLoopMode
                                                      dequeue:YES]) {
            [app sendEvent:event];
        }
        [app updateWindows];
    }
    return std::exchange(Data->PendingEvents, {});
}

void *Window::NativeView() const { return (__bridge void *)Data->View; }

void Window::WarpCursor(float x, float y) {
    const NSPoint window_point{x, Data->View.bounds.size.height - y};
    const NSRect screen_rect = [Data->NativeWindow convertRectToScreen:NSMakeRect(window_point.x, window_point.y, 0, 0)];
    const CGFloat primary_height = NSScreen.screens.firstObject.frame.size.height;
    CGWarpMouseCursorPosition(CGPointMake(screen_rect.origin.x, primary_height - screen_rect.origin.y));
    CGAssociateMouseAndMouseCursorPosition(true);
}

void Window::UpdateDrawableSize() {
    auto *const layer = static_cast<CAMetalLayer *>(Data->View.layer);
    layer.contentsScale = Data->NativeWindow.backingScaleFactor;
    layer.drawableSize = [Data->View convertSizeToBacking:Data->View.bounds.size];
}
} // namespace MacPlatform
