#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "shoaib_har_cnn_quant.h"

// Tensor arena size — tune down/up as needed (start 48k)
constexpr int kTensorArenaSize = 48 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// ---- Make resolver and interpreter static/globals so they outlive setup() ----
static tflite::MicroMutableOpResolver<12> micro_op_resolver; // reserve enough slots
static tflite::MicroInterpreter* tflInterpreter = nullptr;
static TfLiteTensor* tflInputTensor = nullptr;
static TfLiteTensor* tflOutputTensor = nullptr;
static const tflite::Model* tflModel = nullptr;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("Start");

  // Load model from generated header
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch");
    while (1);
  }

  // Register ops — add everything your model needs (example set)
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddExpandDims();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddDequantize();
  micro_op_resolver.AddRelu();           // in case activation uses RELU
  micro_op_resolver.AddSqueeze();        // sometimes appears when reshaping
  micro_op_resolver.AddDepthwiseConv2D();// if depthwise conv is in model
  // --- add other ops your model requires ---

  // Create the interpreter statically
  static tflite::MicroInterpreter static_interpreter(
      tflModel, micro_op_resolver, tensor_arena, kTensorArenaSize);
  tflInterpreter = &static_interpreter;

  Serial.println("Allocating tensors...");
  TfLiteStatus allocate_status = tflInterpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1);
  }
  Serial.println("Tensors allocated");

  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  Serial.print("Input type: ");
  Serial.println(tflInputTensor->type);  // 9 for int8
  Serial.print("Input dims: ");
  for (int i=0;i<tflInputTensor->dims->size;i++) {
    Serial.print(tflInputTensor->dims->data[i]);
    Serial.print(i < tflInputTensor->dims->size-1 ? "x":"\n");
  }

  Serial.println("Setup complete.");
}

void loop() {
  // Use quantization params: scale & zero_point to fill input properly
  const float scale = tflInputTensor->params.scale;        // e.g. 0.0039
  const int zero_point = tflInputTensor->params.zero_point; // e.g. -128

  int n = 1;
  for (int i=0;i<tflInputTensor->dims->size;i++) n *= tflInputTensor->dims->data[i];

  // Fill with pseudo-random quantized values (centered around zero)
  for (int i = 0; i < n; i++) {
    // create a float in [-1, 1], then quantize to int8 using tensor params
    float v = (float)(rand() % 201 - 100) / 100.0f; // -1.0 .. +1.0
    int32_t q = (int32_t)round(v / scale) + zero_point;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    tflInputTensor->data.int8[i] = (int8_t)q;
  }

  Serial.println("Running inference...");
  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  if (invokeStatus != kTfLiteOk) {
    Serial.println("Inference failed!");
    while(1);
  }

  // read outputs (if int8)
  int out0 = tflOutputTensor->dims->data[0];
  int out1 = tflOutputTensor->dims->data[1];
  Serial.println("Output values:");
  for (int i = 0; i < out1; i++) {
    Serial.print("Class ");
    Serial.print(i);
    Serial.print(": ");
    Serial.println(tflOutputTensor->data.int8[i]);
  }

  delay(1000);
}
