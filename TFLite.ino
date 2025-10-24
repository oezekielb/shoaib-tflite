#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "MyBoschSensor.h"
#include "SignalQueue.h"
// #include "shoaib_har_cnn.h"
#include "shoaib_har_lstm.h"


tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

void handle_signal() {
  float ax, ay, az, gx, gy, gz;
  if (myIMU.readAcceleration(ax, ay, az) && myIMU.readGyroscope(gx, gy, gz)) {
    signal_queue.push(ax, ay, az, gx, gy, gz);
  }
}

void predict(float* ax, float* ay, float* az, float* gx, float* gy, float* gz) {
  // copy the signal data to the input tensor
  for (int i = 0; i < WINDOW; i++) {
    tflInputTensor->data.f[i * 6 + 0] = ax[i];
    tflInputTensor->data.f[i * 6 + 1] = ay[i];
    tflInputTensor->data.f[i * 6 + 2] = az[i];
    tflInputTensor->data.f[i * 6 + 3] = gx[i];
    tflInputTensor->data.f[i * 6 + 4] = gy[i];
    tflInputTensor->data.f[i * 6 + 5] = gz[i];
  }

  // invoke the model
  TfLiteStatus invoke_status = tflInterpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // get the prediction results from the output tensor
  int predicted_label = -1;
  float max_score = -1.0f;
  for (int i = 0; i < tflOutputTensor->dims->data[1]; i++) {
    float score = tflOutputTensor->data.f[i];
    if (score > max_score) {
      max_score = score;
      predicted_label = i;
    }
  }

  Serial.print("Predicted label: ");
  Serial.print(predicted_label);
  Serial.print(" (score: ");
  Serial.print(max_score);
  Serial.println(")");
}

void setup() {
  Serial.begin(9600);
  while (!Serial);

  myIMU.debug(Serial);
  myIMU.onInterrupt(handle_signal);
  // initialize the IMU
  if (!myIMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // print out the samples rates of the IMUs
  Serial.print("Accelerometer sample rate = ");
  Serial.print(myIMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate = ");
  Serial.print(myIMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  Serial.println();

  // get the TFL representation of the model byte array
  const tflite::Model* tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }
  
  // Create an interpreter to run the model
  // Register required TFLM operators before creating the interpreter
  Serial.println("Registering TFLM operators...");
  tflite::MicroMutableOpResolver<7> tflOpsResolver;
  tflOpsResolver.AddConv2D();
  tflOpsResolver.AddMaxPool2D();
  tflOpsResolver.AddReshape();
  tflOpsResolver.AddFullyConnected();
  tflOpsResolver.AddSoftmax();
  tflOpsResolver.AddQuantize();
  tflOpsResolver.AddDequantize();

  // Create a static memory buffer for TFLM, the size may need to
  // be adjusted based on the model you are using
  Serial.println("Creating tensor arena...");
  const int tensorArenaSize = 28 * 1024;
  uint8_t  tensorArena[tensorArenaSize];

  Serial.println("Creating interpreter...");
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize);

  // Allocate memory for the model's input and output tensors
  Serial.println("Allocating tensors...");
  tflInterpreter->AllocateTensors();

  // // Get pointers for the model's input and output tensors
  // Serial.println("Getting input and output tensors...");
  // tflInputTensor = tflInterpreter->input(0);
  // tflOutputTensor = tflInterpreter->output(0);

  Serial.println("Setup complete.");
}

float ax[WINDOW], ay[WINDOW], az[WINDOW], gx[WINDOW], gy[WINDOW], gz[WINDOW];
void loop() {
  // uint size = signal_queue.get_size();
  // Serial.print("Queue size: ");
  // Serial.println(size);

  // while (signal_queue.is_full()) {
  //   signal_queue.peek(ax, ay, az, gx, gy, gz);

  //   Serial.println("Making prediction...");
  //   predict(ax, ay, az, gx, gy, gz); 
  // }
}
