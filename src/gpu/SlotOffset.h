#pragma once

#include "gpu/Types.h"

struct SlotOffset {
    uint32_t Slot DEFAULT(InvalidSlot);
    uint32_t Offset DEFAULT();
};

// A pass's record at Job in its storage, and a pass parameter such as a radix shift, a round's buffer parity or a stage.
struct JobPushConstants {
    SlotOffset Job DEFAULT();
    uint32_t Parameter DEFAULT();
};
