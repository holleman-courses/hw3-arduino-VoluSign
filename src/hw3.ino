#include <Arduino.h>
#include <TensorFlowLite.h>
#include "sine_model_data.h"  // Include quantized TFLite model as C byte array
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// TensorFlow Lite globals
constexpr int kTensorArenaSize = 136 * 1024; 
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Quantization parameters (original model)
const float input_scale = 0.007831485942006111; 
const int input_zero_point = -1;                
const float output_scale = 0.017527734860777855; 
const int output_zero_point = 31;               

void setup() {
    delay.5000()
    Serial.begin(115200);
    while (!Serial) {} // Wait for Serial monitor

    Serial.println("Initializing model.");

    // Load TFLite model
    model = tflite::GetModel(sine_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model version mismatch!");
        return;
    }

    // Instantiate interpreter
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);
    interpreter = &static_interpreter;

    // Allocate memory for tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Failed to allocate tensors!");
        return;
    }

    // Get input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
    Serial.println("Enter 7 comma-separated integers in the range [-128, 127]:");
}

void loop() {
    if (Serial.available() > 0) {
        String user_input = Serial.readStringUntil('\n');
        user_input.trim();

        int values[7];
        int count = 0;
        char* token = strtok((char*)user_input.c_str(), ",");
        while (token != nullptr && count < 7) {
            values[count++] = atoi(token);
            token = strtok(nullptr, ",");
        }

        // Validate input
        if (count != 7) {
            Serial.println("Error: Enter 7 integers.");
            return;
        }

        // Measure USB print time
        unsigned long t0 = micros();
        Serial.println("Processing...");
        unsigned long t1 = micros();

        // Load values into input tensor
        for (int i = 0; i < 7; i++) {
            // Verify inputs are in range
            if (values[i] < -128 || values[i] > 127) {
                Serial.println("Error: Input values must be in the range [-128, 127].");
                return;
            }
            // Scale int8 inputs to [-1, 1]
            float scaled_input = (values[i] - input_zero_point) * input_scale;
            input->data.f[i] = scaled_input; 
        }

        // Run inference and measure time
        if (interpreter->Invoke() != kTfLiteOk) {
            Serial.println("Model inference failed.");
            return;
        }
        unsigned long t2 = micros();

        // Get the predicted output (float32)
        float dequantized_output = output->data.f[0];

        // Scale the output back to int8
        int8_t quantized_output = static_cast<int8_t>((dequantized_output / output_scale) + output_zero_point);

        // Print results
        Serial.print("Quantized Output: ");
        Serial.println(quantized_output);

        // Print timing results
        Serial.print("Printing time = ");
        Serial.print(t1 - t0);
        Serial.print(" us.  Inference time = ");
        Serial.print(t2 - t1);
        Serial.println(" us.");
    }
}