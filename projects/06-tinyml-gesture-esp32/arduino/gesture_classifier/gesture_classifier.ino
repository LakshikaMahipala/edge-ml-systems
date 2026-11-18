// TinyML gesture classifier for Arduino Nano ESP32 (TensorFlow Lite Micro).
//
// Loads the INT8 TCN model from model.h, runs it on the test window in
// gesture_sample.cpp, and prints the predicted gesture, confidence and latency.
//
// Library: "TensorFlowLite_ESP32" (Arduino Library Manager).
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "gesture_sample.h"
#include "labels.h"
#include "model.h"

namespace {
constexpr int kLedPin = LED_BUILTIN;
// Working memory for activations. The model input alone is 50 x 250 values;
// increase this if AllocateTensors() reports that the arena is too small.
constexpr int kArenaSize = 160 * 1024;
alignas(16) uint8_t tensor_arena[kArenaSize];

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
bool ready = false;
}  // namespace

// Writes one float into the input tensor, quantizing if the model is INT8.
static void set_input(int i, float v) {
  if (input->type == kTfLiteInt8) {
    int q = static_cast<int>(roundf(v / input->params.scale)) + input->params.zero_point;
    input->data.int8[i] = static_cast<int8_t>(constrain(q, -128, 127));
  } else {
    input->data.f[i] = v;
  }
}

static float get_output(int i) {
  if (output->type == kTfLiteInt8)
    return (output->data.int8[i] - output->params.zero_point) * output->params.scale;
  return output->data.f[i];
}

void setup() {
  Serial.begin(115200);
  pinMode(kLedPin, OUTPUT);

  const tflite::Model* model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version does not match the library.");
    return;
  }
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena,
                                                     kArenaSize, nullptr);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed: increase kArenaSize.");
    return;
  }
  input = interpreter->input(0);
  output = interpreter->output(0);

  const int n_in = input->bytes / (input->type == kTfLiteInt8 ? 1 : sizeof(float));
  Serial.printf("Model input: %d values (%s), sample: %d values, arena used: %u bytes\n",
                n_in, input->type == kTfLiteInt8 ? "int8" : "float32", kSampleLength,
                static_cast<unsigned>(interpreter->arena_used_bytes()));
  if (n_in != kSampleLength) {
    Serial.println("ERROR: test sample size does not match the model input. "
                   "Re-export gesture_sample.cpp from the training notebook.");
    return;
  }
  ready = true;
}

void loop() {
  if (!ready) { delay(1000); return; }

  for (int i = 0; i < kSampleLength; i++) set_input(i, kGestureSample[i]);

  const uint32_t t0 = micros();
  const TfLiteStatus status = interpreter->Invoke();
  const uint32_t latency_us = micros() - t0;
  if (status != kTfLiteOk) { Serial.println("Invoke() failed"); delay(1000); return; }

  int best = 0;
  for (int i = 1; i < kNumGestures; i++)
    if (get_output(i) > get_output(best)) best = i;

  Serial.printf("Predicted: %s (score %.2f) | inference %.1f ms\n", kGestureLabels[best],
                get_output(best), latency_us / 1000.0f);
  digitalWrite(kLedPin, HIGH);
  delay(200);
  digitalWrite(kLedPin, LOW);
  delay(2000);
}
