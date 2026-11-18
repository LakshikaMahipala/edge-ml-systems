TinyML Micro 

Goal
- Scaffold a TFLite Micro / CMSIS-NN style deployment.
- Provide an operator cost model for tiny CNNs.

Docs
- docs/tflm_quickstart.md
- docs/cmsis_nn_what_and_why.md
- docs/tiny_cnn_cost_model.md
- model/tiny_cnn_spec.md

Scripts
- scripts/op_cost_estimator.py

C++ skeleton
- cxx/ contains a placeholder micro inference wrapper.
Later we will replace model_data.h with real .tflite bytes and link TFLM.

Run later
python scripts/op_cost_estimator.py --alpha 3.0
