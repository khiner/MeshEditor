#pragma once

#include "faust/gui/MetaDataUI.h"
#include "faust/gui/PathBuilder.h"
#include "faust/gui/UI.h"

#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

using Real = float;

enum ItemType {
    ItemType_None = 0,
    // Containers
    ItemType_HGroup,
    ItemType_VGroup,
    ItemType_TGroup,

    // Widgets
    ItemType_Button,
    ItemType_CheckButton,
    ItemType_VSlider,
    ItemType_HSlider,
    ItemType_NumEntry,
    ItemType_HBargraph,
    ItemType_VBargraph,

    // Types specified with metadata
    ItemType_Knob,
    ItemType_Menu,
    ItemType_VRadioButtons,
    ItemType_HRadioButtons,
};

// Label, shortname, or complete path (to discriminate between possibly identical labels
// at different locations in the UI hierarchy) can be used to access any created widget.
// See Faust's `APIUI` for possible extensions (response curves, gyro, ...).
class FaustParams : public UI, public MetaDataUI, public PathBuilder {
public:
    FaustParams() = default;
    ~FaustParams() override = default;

    struct Item {
        Item(const ItemType type, std::string_view label, Real *zone = nullptr, Real min = 0, Real max = 0, Real init = 0, Real step = 0, const char *tooltip = nullptr, std::vector<Item> items = {})
            : type(type), id(label), label(label == "0x00" ? "" : label), zone(zone), min(min), max(max), init(init), step(step), tooltip(tooltip), items(std::move(items)) {}

        const ItemType type{ItemType_None};
        const std::string id, label; // `id` will be the same as `label` unless it's the special empty group label of '0x00', in which case `label` will be empty.
        Real *zone; // Only meaningful for widget items (not container items)
        const Real min, max; // Only meaningful for sliders, num-entries, and bar graphs.
        const Real init, step; // Only meaningful for sliders and num-entries.
        const char *tooltip;
        std::vector<Item> items; // Only populated for container items (groups)
        bool logscale{false};
    };
    struct NamesAndValues {
        std::vector<std::string> names{};
        std::vector<double> values{};
    };

    void openHorizontalBox(const char *label) override {
        pushLabel(label);
        activeGroup().items.emplace_back(ItemType_HGroup, label);
        Groups.push(&activeGroup().items.back());
    }
    void openVerticalBox(const char *label) override {
        pushLabel(label);
        activeGroup().items.emplace_back(ItemType_VGroup, label);
        Groups.push(&activeGroup().items.back());
    }
    void openTabBox(const char *label) override {
        pushLabel(label);
        activeGroup().items.emplace_back(ItemType_TGroup, label);
        Groups.push(&activeGroup().items.back());
    }
    void closeBox() override {
        Groups.pop();
        if (popLabel()) {
            computeShortNames();
        }
    }

    // Active widgets
    void addButton(const char *label, Real *zone) override {
        addUiItem(ItemType_Button, label, zone);
    }
    void addCheckButton(const char *label, Real *zone) override {
        addUiItem(ItemType_CheckButton, label, zone);
    }
    void addHorizontalSlider(const char *label, Real *zone, Real init, Real min, Real max, Real step) override {
        addSlider(label, zone, init, min, max, step, false);
    }
    void addVerticalSlider(const char *label, Real *zone, Real init, Real min, Real max, Real step) override {
        addSlider(label, zone, init, min, max, step, true);
    }
    void addNumEntry(const char *label, Real *zone, Real init, Real min, Real max, Real step) override {
        addUiItem(ItemType_NumEntry, label, zone, min, max, init, step);
    }
    void addSlider(const char *label, Real *zone, Real init, Real min, Real max, Real step, bool is_vertical) {
        if (isKnob(zone)) addUiItem(ItemType_Knob, label, zone, min, max, init, step);
        else if (isRadio(zone)) addRadioButtons(label, zone, init, min, max, step, fRadioDescription[zone].c_str(), is_vertical);
        else if (isMenu(zone)) addMenu(label, zone, init, min, max, step, fMenuDescription[zone].c_str());
        else addUiItem(is_vertical ? ItemType_VSlider : ItemType_HSlider, label, zone, min, max, init, step);
    }
    void addRadioButtons(const char *label, Real *zone, Real init, Real min, Real max, Real step, const char *text, bool is_vertical) {
        NamesForZone[zone] = {};
        ValuesForZone[zone] = {};
        parseMenuList(text, NamesForZone.at(zone), ValuesForZone.at(zone));
        addUiItem(is_vertical ? ItemType_VRadioButtons : ItemType_HRadioButtons, label, zone, min, max, init, step);
    }
    void addMenu(const char *label, Real *zone, Real init, Real min, Real max, Real step, const char *text) {
        NamesForZone[zone] = {};
        ValuesForZone[zone] = {};
        parseMenuList(text, NamesForZone.at(zone), ValuesForZone.at(zone));
        addUiItem(ItemType_Menu, label, zone, min, max, init, step);
    }

    // Passive widgets
    void addHorizontalBargraph(const char *label, Real *zone, Real min, Real max) override {
        addUiItem(ItemType_HBargraph, label, zone, min, max);
    }
    void addVerticalBargraph(const char *label, Real *zone, Real min, Real max) override {
        addUiItem(ItemType_VBargraph, label, zone, min, max);
    }

    // Soundfile
    void addSoundfile(const char *, const char *, Soundfile **) override {}

    // Metadata declaration
    void declare(Real *zone, const char *key, const char *value) override {
        MetaDataUI::declare(zone, key, value);
    }

    Real *getZoneForLabel(const char *label) {
        return ZoneForLabel.contains(label) ? ZoneForLabel.at(label) : nullptr;
    }

    Item Root{ItemType_None, ""};
    std::unordered_map<const Real *, std::vector<std::string>> NamesForZone;
    std::unordered_map<const Real *, std::vector<double>> ValuesForZone;

    void DrawItem(const Item &);
    void Draw() { DrawItem(Root); }

private:
    void addUiItem(const ItemType type, const char *label, Real *zone, Real min = 0, Real max = 0, Real init = 0, Real step = 0) {
        Item item{type, label, zone, min, max, init, step, fTooltip.contains(zone) ? fTooltip.at(zone).c_str() : nullptr};
        if (fLogSet.contains(zone)) item.logscale = true;
        activeGroup().items.emplace_back(std::move(item));
        fFullPaths.push_back(buildPath(label));
        ZoneForLabel[label] = zone;
    }

    Item &activeGroup() { return Groups.empty() ? Root : *Groups.top(); }

    std::stack<Item *> Groups{};
    std::unordered_map<std::string, Real *> ZoneForLabel{};
};
