#include "network.h"
#include "tanh_lut.h"
#define SCALE_FACTOR 511

l_quantized_type LeakyReLU(l_quantized_type res) {
    if (res < 0)
        return (l_quantized_type)(res * (l_quantized_type)0.2);
    return res;
}

l_quantized_type tanh_lut_func(l_quantized_type res) {
    if (res >= 2) return 1;
    if (res < -2) return -1;
    ap_int<BITS+2> i = res.range();
    return tanh_vals[(BITS_EXP/2) + i.to_int()];
}

extern "C" {
void network(
    float *x,
    float *y,
    short *fc1_weights,
    short *fc2_weights,
    short *fc3_weights,
    short *fc4_weights,
    float *fc1_biases,
    float *fc2_biases,
    float *fc3_biases,
    float *fc4_biases,
    int INPUT_IMAGES
)
{
    // ── AXI Interfaces ──────────────────────────────────────
    #pragma HLS INTERFACE m_axi port=x            bundle=gmem0 \
        depth=6145 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=y            bundle=gmem1 \
        depth=6144 max_write_burst_length=256
    #pragma HLS INTERFACE m_axi port=fc1_weights  bundle=gmem2 \
        depth=3146240 max_read_burst_length=256 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=fc2_weights  bundle=gmem3 \
        depth=262144 max_read_burst_length=256 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=fc3_weights  bundle=gmem4 \
        depth=262144 max_read_burst_length=256 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=fc4_weights  bundle=gmem5 \
        depth=3145728 max_read_burst_length=256 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=fc1_biases   bundle=gmem6 depth=512
    #pragma HLS INTERFACE m_axi port=fc2_biases   bundle=gmem7 depth=512
    #pragma HLS INTERFACE m_axi port=fc3_biases   bundle=gmem8 depth=512
    #pragma HLS INTERFACE m_axi port=fc4_biases   bundle=gmem9 depth=6144
    
    #pragma HLS INTERFACE s_axilite port=x
    #pragma HLS INTERFACE s_axilite port=y
    #pragma HLS INTERFACE s_axilite port=fc1_weights
    #pragma HLS INTERFACE s_axilite port=fc2_weights
    #pragma HLS INTERFACE s_axilite port=fc3_weights
    #pragma HLS INTERFACE s_axilite port=fc4_weights
    #pragma HLS INTERFACE s_axilite port=fc1_biases
    #pragma HLS INTERFACE s_axilite port=fc2_biases
    #pragma HLS INTERFACE s_axilite port=fc3_biases
    #pragma HLS INTERFACE s_axilite port=fc4_biases
    #pragma HLS INTERFACE s_axilite port=INPUT_IMAGES
    #pragma HLS INTERFACE s_axilite port=return

    // Pre-compute reciprocal (sekali, bukan jutaan kali)
    const float INV_SCALE = 1.0f / SCALE_FACTOR;
    
    // Buffer lokal di BRAM
    quantized_type   xbuf[N1];
    l_quantized_type layer_1_out[M1];
    l_quantized_type layer_2_out[M2];
    l_quantized_type layer_3_out[M3];
    
    // Bias cache di BRAM
    float bias1_cache[M1];
    float bias2_cache[M2];
    float bias3_cache[M3];
    float bias4_cache[M4];

    for (int iter = 0; iter < INPUT_IMAGES; iter++) {
        
        // Load input
        read_input:
        for (int i = 0; i < N1; i++) {
            #pragma HLS PIPELINE II=1
            xbuf[i] = x[iter*N1 + i];
        }

        // Cache biases (akses cepat di MAC)
        cache_bias1:
        for (int i = 0; i < M1; i++) {
            #pragma HLS PIPELINE II=1
            bias1_cache[i] = fc1_biases[i];
        }
        cache_bias2:
        for (int i = 0; i < M2; i++) {
            #pragma HLS PIPELINE II=1
            bias2_cache[i] = fc2_biases[i];
        }
        cache_bias3:
        for (int i = 0; i < M3; i++) {
            #pragma HLS PIPELINE II=1
            bias3_cache[i] = fc3_biases[i];
        }
        cache_bias4:
        for (int i = 0; i < M4; i++) {
            #pragma HLS PIPELINE II=1
            bias4_cache[i] = fc4_biases[i];
        }

        //  Layer 1 — pakai INV_SCALE (perkalian, bukan pembagian)
        layer_1:
        for (int j = 0; j < M1; j++) {
            l_quantized_type result = (l_quantized_type)bias1_cache[j];
            for (int i = 0; i < N1; i++) {
                #pragma HLS PIPELINE II=1
                result += xbuf[i] * 
                    (quantized_type)(fc1_weights[j*N1 + i] * INV_SCALE);
            }
            layer_1_out[j] = LeakyReLU(result);
        }

        //  Layer 2 — pakai INV_SCALE
        layer_2:
        for (int i = 0; i < M2; i++) {
            l_quantized_type result = (l_quantized_type)bias2_cache[i];
            for (int j = 0; j < N2; j++) {
                #pragma HLS PIPELINE II=1
                result += layer_1_out[j] * 
                    (quantized_type)(fc2_weights[i*N2 + j] * INV_SCALE);
            }
            layer_2_out[i] = LeakyReLU(result);
        }

        //  Layer 3 — pakai INV_SCALE
        layer_3:
        for (int i = 0; i < M3; i++) {
            l_quantized_type result = (l_quantized_type)bias3_cache[i];
            for (int j = 0; j < N3; j++) {
                #pragma HLS PIPELINE II=1
                result += layer_2_out[j] * 
                    (quantized_type)(fc3_weights[i*N3 + j] * INV_SCALE);
            }
            layer_3_out[i] = LeakyReLU(result);
        }

        // Layer 4 + Tanh — pakai INV_SCALE
        layer_4:
        for (int i = 0; i < M4; i++) {
            l_quantized_type result = (l_quantized_type)bias4_cache[i];
            for (int j = 0; j < N4; j++) {
                #pragma HLS PIPELINE II=1
                result += layer_3_out[j] * 
                    (quantized_type)(fc4_weights[i*N4 + j] * INV_SCALE);
            }
            y[iter*M4 + i] = tanh_lut_func(result).to_float();
        }
    }
}
}