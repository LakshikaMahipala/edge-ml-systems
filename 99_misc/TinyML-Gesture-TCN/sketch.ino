#include "gesture_sample.h"
#include "model.h"
#include "labels.h"

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// LED pin (GPIO 2 for ESP32 in Wokwi)
#define SOS_LED 2

// TFLite tensor arena
constexpr int kArenaSize = 10 * 1024;
uint8_t tensor_arena[kArenaSize];

// Interpreter and tensors
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

void setup() {
  Serial.begin(115200);
  pinMode(SOS_LED, OUTPUT);

  // Load model
  const tflite::Model* model = tflite::GetModel(model_tflite);
  static tflite::AllOpsResolver resolver;

  // Fixed: Updated MicroInterpreter constructor with 5 args
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kArenaSize, nullptr
  );
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();

  input = interpreter->input(0);
  output = interpreter->output(0);
}

void run_inference(float* input_data) {
  for (int i = 0; i < 128 * 6; i++) {
    input->data.f[i] = input_data[i];
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // Find gesture with max confidence
  int max_index = 0;
  float max_score = output->data.f[0];
  for (int i = 1; i < output->dims->data[1]; i++) {
    if (output->data.f[i] > max_score) {
      max_index = i;
      max_score = output->data.f[i];
    }
  }

  Serial.print("Detected gesture: ");
  Serial.println(gesture_labels[max_index]);

  // Blink LED if SOS
  if (strcmp(gesture_labels[max_index], "sos") == 0) {
    digitalWrite(SOS_LED, HIGH);
    delay(300);
    digitalWrite(SOS_LED, LOW);
    delay(300);
  } else {
    digitalWrite(SOS_LED, LOW);
  }
}

void loop() {
  Serial.println("Running inference on sample...");

  // Fixed: Only one sample is used now, access it as 2D array
  float* flat_input = &input_data[0][0];
  run_inference(flat_input);

  delay(3000);
}
