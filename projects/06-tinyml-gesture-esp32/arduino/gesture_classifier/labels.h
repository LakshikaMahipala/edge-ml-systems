#pragma once
// Same order as sklearn LabelEncoder in the training notebook (alphabetical).
static const char* const kGestureLabels[] = {
    "Clapping", "Fist Making", "Index Thumb Tap", "Thumb Up", "Wrist Extension"};
static const int kNumGestures = sizeof(kGestureLabels) / sizeof(kGestureLabels[0]);
