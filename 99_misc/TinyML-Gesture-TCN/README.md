# TinyML Gesture Recognition with Arduino Nano ESP32

This project is a complete TinyML Study  from training a gesture recognition model 🧠 to running it in real-time on a tiny Arduino Nano ESP32 board 🔌. It is an AI model  that inferences on an actual microcontroller without needing the cloud or internet.  
Used data from motion sensors (accelerometer + gyroscope) to train a model that can recognize 5 different human gestures:

👏 Clapping

✊ Fist Making

👉 Index Thumb Tap

👍 Thumb Up

↗️ Wrist Extension

---
# Dataset
**Used a public dataset called HGAG-DATA1: [Link to Dataset](https://plu.mx/plum/a?mendeley_data_id=mkhn7kxjvy&theme=plum-bigben-theme)**  
The dataset we used is called HGAG-DATA1, a rich human gesture dataset collected from 43 healthy individuals aged between 18 and 69. The participants include both athletic and non-athletic individuals, as well as a mix of right-handed and left-handed people. Each participant was instructed to perform 11 different everyday hand gestures, such as clapping, fist making, wrist movements, and thumb gestures. For each gesture, they performed 50 repetitions, resulting in a total of over 23,000 labeled gesture recordings.  
Each gesture was recorded using a 6-axis motion sensor setup, combining a 3-axis accelerometer and a 3-axis gyroscope, giving us a rich time-series dataset with six features per timestep.

However, since our project focuses on deploying the final model on a microcontroller (Arduino Nano ESP32), we intentionally used only a subset of the data to keep things lightweight and fast. Specifically, we selected 5 gestures from the full set, included recordings from all 43 subjects, and only used the .csv files that contain the actual sensor readings. We then cleaned, padded, and normalized the data to ensure consistency across samples.  
The result? A small but well-balanced dataset that's ideal for training a compact deep learning model—one that can still handle real-world motion input with impressive accuracy while fitting on a tiny embedded device.

---
# TCN (Temporal Convolutional Network)

A TCN (Temporal Convolutional Network) model is trained with a supercharged 1D CNN built specially for sequence data — like motion, audio, or text — where order matters.

TCNs work by sliding filters across time steps (just like how CNNs slide filters across image pixels), but they have two cool tricks:

🔄 **1. Causal Convolution**  
This means the output at time t only sees inputs from time ≤ t. No peeking into the future!

🧱 **2. Dilated Convolution**  
Dilated filters can skip time steps like every 1st, 2nd, 4th, 8th… this allows the model to see a much wider window of time without growing huge.

---

### 🧱 Building TCN Block

Here’s a simplified version of what each TCN block does:

```python
def residual_tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.2):
    shortcut = x  # Save input for skip connection

    x = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(dropout_rate)(x)

    x = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)

    # Squeeze-and-Excite: helps model focus on important features
    x = se_block(x)

    x = SpatialDropout1D(dropout_rate)(x)

    # Match shapes if needed
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)

    x = Add()([x, shortcut])  # Residual connection
    return Activation('relu')(x)
```

---

### 💪 Squeeze-and-Excitation (SE) Block

Why SE? Because not all features are equally useful.

```python
def se_block(x, ratio=8):
    filters = x.shape[-1]
    se = GlobalAveragePooling1D()(x)
    se = Reshape((1, filters))(se)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    return multiply([x, se])
```

This learns to "excite" the important channels and suppress the unimportant ones.

## 🧠 Final Model Architecture

Let’s stack these blocks:

```python
def build_optimized_tcn(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = inputs

    for d in [1, 2, 4, 8, 16]:  # exponentially increasing dilation
        x = residual_tcn_block(x, filters=64, kernel_size=3, dilation_rate=d)

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)
```

This gives us a powerful yet lightweight model, perfect for real-time gesture detection on microcontrollers.

---

## 🎯 Loss Function: Focal Loss

To handle imbalanced or noisy data, we used Focal Loss instead of regular cross-entropy.

```python
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(weight * cross_entropy)
    return loss
```

---

## 🛠️ Model Compilation & Training

After defining the model, we compile it using the Adam optimizer and our custom focal loss function:

```python
model = build_optimized_tcn(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)

model.summary()
```

This tells Keras:  
📉 Use focal_loss to focus on hard-to-classify gestures.  
📊 Track accuracy as a metric.  
🔧 Adjust weights using Adam optimizer with a moderate learning rate.

---

## 📈 Training with Early Stopping and Checkpoints

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best_tcn_model.h5", save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)
```

EarlyStopping stops training if the model stops improving to avoid overfitting.  
ModelCheckpoint saves the best model during training based on validation loss.

---

## 📊 Training & Validation Curves

To visualize how well the model is learning:

```python
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Accuracy / Loss")
plt.legend()
plt.grid()
plt.title("📉 Training Progress")
plt.show()
```

This helps us see:  
Are we overfitting?  
Is the model still improving?  
Did validation accuracy plateau?

---

## ✅ Evaluate Model with Confusion Matrix

After training, we can see how good the model is for each gesture using a confusion matrix:

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Predict on validation set
y_pred = model.predict(X_val)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_val, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("🧩 Confusion Matrix")
plt.show()
```

---

## 🧮 Per-Class Precision & Recall

This gives more insight than just accuracy:

```python
print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))
```

You’ll see something like:

```plaintext
              precision    recall  f1-score   support

     Clapping       0.74      0.80      0.77       45
  Fist Making       0.71      0.67      0.69       42
Index Thumb Tap     0.68      0.62      0.65       44
     Thumb Up       0.70      0.75      0.72       41
Wrist Extension     0.73      0.69      0.71       40
```

**Precision**: How many predicted were actually correct.  
**Recall**: How many actual were detected.  
**F1-score**: Harmonic mean of precision and recall.

---

## 🧳 Exporting TCN Model to TensorFlow Lite (TFLite)

Once the model was trained and performing decently, it was time to shrink it down and get it ready for deployment on the Arduino Nano ESP32.

### 🎯 Why TFLite?

TinyML models need to be small.  
TFLite allows compressing, quantizing, and optimizing models for microcontrollers.

---

## 🧠 Convert to TFLite

Used TensorFlow Lite’s converter to compress the Keras model:

```python
import tensorflow as tf

# Load best saved model
model = tf.keras.models.load_model("best_tcn_model.h5", custom_objects={'loss': focal_loss()})

# Convert to TFLite (floating point)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open("model_float.tflite", "wb") as f:
    f.write(tflite_model)
```

---

## 🔬 Quantize the Model (INT8 for Arduino)

Need to shrink the model size and make it efficient for embedded hardware:

```python
# Enable optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset for calibration
def rep_data():
    for sample in X_train[:100]:
        yield [sample.astype(np.float32)]

converter.representative_dataset = rep_data
converter.target_spec.supported_types = [tf.float16]  # Optional: tf.int8 if needed

quant_model = converter.convert()

# Save quantized model
with open("model.tflite", "wb") as f:
    f.write(quant_model)
```

This generates a lightweight `.tflite` model file (usually < 100 KB).

---

## 🧾 Convert .tflite to C Header (`model.h`)

Now embed the model directly into Arduino by converting it to a `.h` file:

### ✅ Option 1: Use Python

```bash
xxd -i model.tflite > model.h
```

**Or:**

```python
with open("model.tflite", "rb") as f:
    hex_array = f.read()

with open("model.h", "w") as out:
    out.write("const unsigned char model_tflite[] = {\n")
    out.write(",\n".join(["  " + ", ".join(f"0x{b:02x}" for b in hex_array[i:i+12])
                          for i in range(0, len(hex_array), 12)]))
    out.write("\n};\n")
    out.write(f"const int model_len = {len(hex_array)};\n")
```
## ✍️ Preparing Test Data for Arduino Inference

After training the model, it is needed to export a real gesture sample from the dataset for inference testing.  
Instead of dumping everything into a `.cpp` file, we followed best Arduino coding practices by separating the data into two files:

---

### ✅ gesture_sample.h (Header File)

This file acts as a header and declares the data. This is a declaration file. It tells the Arduino sketch there’s a float array called `input_data` and we have just 1 test sample.  
Here’s what it contains:

```cpp
#ifndef GESTURE_SAMPLE_H
#define GESTURE_SAMPLE_H

extern float input_data[128][6];  // Declare the data exists
#define NUM_SAMPLES 1             // Total test samples

#endif
```

So the compiler will successfully use `input_data` in your `.ino` file.

---

### ✅ gesture_sample.cpp (Source File)

This file holds the actual data values — a 2D array of shape `[128][6]`. Each of the 128 rows is a single timestep (sensor reading), and the 6 values are:

- 3 Accelerometer axes (X, Y, Z)  
- 3 Gyroscope axes (X, Y, Z)

So, 128 sensor readings → each with 6 values = 768 input features per gesture.  
Example structure:

```cpp
#include "gesture_sample.h"

float input_data[128][6] = {
  {0.028392, 0.005251, 0.042692, 0.006808, 0.017894, -0.008793},
  {1.626350, 0.949168, -1.376201, -0.495231, 0.218532, 0.151405},
  ...
};
```

Has created this array from a real validation sample that was padded, normalized, and saved directly from Python (using `flatten()` and file writing logic).

---

## 🧪 Running Inference on Arduino

Now that we’ve got our model (`model.h`) and test sample (`gesture_sample.h` + `gesture_sample.cpp`), let’s see how the Arduino sketch actually runs the AI model on the board and makes a decision.

---

### 🧱 Files Working Together

| File              | What It Does |
|-------------------|--------------|
| `model.h`         | Contains the quantized TFLite model as a C byte array (`model_tflite[]`) |
| `gesture_sample.h`| Declares the `input_data` array and the number of samples (`NUM_SAMPLES`) |
| `gesture_sample.cpp` | Defines the actual `input_data[128][6]` float array |
| `labels.h`        | Stores a list of gesture label strings (like `"clapping"`, `"sos"`...) |
| `sketch.ino`      | The Arduino program that loads the model, runs the inference, prints result, and blinks an LED if the gesture is SOS |

---

### 🔧 `setup()`

This function runs once when the board powers on:

```cpp
void setup() {
  Serial.begin(115200);              // Start serial monitor
  pinMode(SOS_LED, OUTPUT);          // Set up LED pin

  // Load and parse the model from model_tflite array
  const tflite::Model* model = tflite::GetModel(model_tflite);

  // Initialize operator support (AllOpsResolver loads all built-in TFLite ops)
  static tflite::AllOpsResolver resolver;

  // Create interpreter with model, op resolver, and memory arena
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kArenaSize, nullptr);
  interpreter = &static_interpreter;

  // Allocate memory for tensors
  interpreter->AllocateTensors();

  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
}
```

Key concept: We’re loading a pre-trained TFLite model into a tiny memory buffer (`tensor_arena`) and setting it up for inference. This works even on low-power boards like the Arduino Nano ESP32!

---

### 🚀 `run_inference()`

This function runs the model on a sample and prints the result:

```cpp
void run_inference(float* input_data) {
  for (int i = 0; i < 128 * 6; i++) {
    input->data.f[i] = input_data[i];  // Copy sample into model input tensor
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // Find the label index with the highest score
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

  // If the gesture is "sos", blink the LED!
  if (strcmp(gesture_labels[max_index], "sos") == 0) {
    digitalWrite(SOS_LED, HIGH);
    delay(300);
    digitalWrite(SOS_LED, LOW);
    delay(300);
  } else {
    digitalWrite(SOS_LED, LOW);
  }
}
```

Key learning: This function manually loads sensor data into the model, runs inference using `interpreter->Invoke()`, and gets predictions from the output tensor.  
It even maps the highest-probability output index back to the label (like `"fist making"`).

---

### 🔁 `loop()`

This is Arduino’s forever-running function.

```cpp
void loop() {
  Serial.println("Running inference on sample...");
  
  // Run on the first (and only) sample we included
  float* flat_input = &input_data[0][0];
  run_inference(flat_input);

  delay(3000);  // Wait before running again
}
```

We're only using one static test gesture sample, but this loop could easily be adapted to:

- Read live data from an IMU (like MPU6050)  
- Stream from Bluetooth  
- Or use multiple gestures  

---

And that's it! The model now runs live on the ESP32, detects a gesture.
