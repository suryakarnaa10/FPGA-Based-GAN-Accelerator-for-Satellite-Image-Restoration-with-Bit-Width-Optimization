#ifndef _LAYER_H_ //jika ada layer H maka langkahi, jika tidak ada maka baca
#define _LAYER_H_

#include <stdlib.h>
#include "ap_fixed.h" //lib fix point

//dimensi network
#define N1 6145 //input  (32×64×3 + 1 flag = 6145)
#define M1 512 //hidden layer 1
#define N2 512 //input layer 2
#define M2 512 //hidden layer 2
#define N3 512 //input layer 3
#define M3 512 //hidden layer 3
#define N4 512 //input layer 4
#define M4 6144 //output  (32×64×3 = 6144)

//bit width
#define BITS 10 // 10 BIT KUANTISASI
#define BITS_EXP 4096

//Tipe data fixed point
typedef ap_fixed<BITS+2, 2, AP_RND> quantized_type;
//ap_fixed<12, 2> untuk 10-bit
//range: -2.0 sampai +1.9995

typedef ap_fixed<BITS+14, 14, AP_RND> l_quantized_type;
//ap_fixed<22, 19> untuk akumulasi
//lebih lebar untuk cegah overflow

#endif //tutup block