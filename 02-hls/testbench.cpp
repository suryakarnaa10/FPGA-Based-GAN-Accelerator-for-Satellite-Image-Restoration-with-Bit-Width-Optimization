#include <iostream>
#include "network.h"
#include "weights_10bit.h"   // ← include di sini saja

extern "C" void network(
    float *x, float *y,
    short *fc1_weights, short *fc2_weights, 
    short *fc3_weights, short *fc4_weights,
    float *fc1_biases, float *fc2_biases, 
    float *fc3_biases, float *fc4_biases,
    int INPUT_IMAGES
);

int main() {
    float x[N1];
    float y[M4];
    
    for (int i = 0; i < N1; i++) x[i] = 0.5f;

    network(
        x, y,
        (short*)fc1_weights, (short*)fc2_weights,
        (short*)fc3_weights, (short*)fc4_weights,
        fc1_biases, fc2_biases, fc3_biases, fc4_biases,
        1
    );

    bool valid = true;
    for (int i = 0; i < M4; i++) {
        if (y[i] < -1.0f || y[i] > 1.0f) {
            std::cout << "ERROR: output[" << i << "] = " << y[i] << std::endl;
            valid = false;
        }
    }

    if (valid) {
        std::cout << "PASS: semua output dalam range [-1,1]" << std::endl;
        std::cout << "Output pertama: " << y[0] << std::endl;
        std::cout << "Output terakhir: " << y[M4-1] << std::endl;
    }
    return 0;
}