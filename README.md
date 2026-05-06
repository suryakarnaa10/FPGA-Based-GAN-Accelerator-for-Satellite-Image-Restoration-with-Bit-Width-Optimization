# FPGA-Based GAN Accelerator for Satellite Image Restoration with Bit-Width Optimization

A hardware-accelerated implementation of a Generative Adversarial Network (GAN) for restoring corrupted satellite imagery, deployed on the AMD Kria KV260 Vision AI Starter Kit. This project explores bit-width quantization trade-offs for neural network accelerators on edge FPGAs, enabling efficient on-board satellite image processing.

---

## The Story Behind This Project

Satellite imagery is essential for agriculture monitoring, disaster response, and environmental research, but raw data captured by satellites often arrives corrupted — sensor failures cause missing blocks, atmospheric interference adds noise, and clouds occlude critical regions. Traditional restoration runs on cloud servers, requiring expensive bandwidth to downlink corrupted images, process them, and uplink results. For time-sensitive applications like disaster response, this latency is unacceptable.

This project asks a simple question: **can we restore satellite imagery directly on edge hardware?** The answer required taking a Generative Adversarial Network trained in PyTorch and translating it to run on a low-power FPGA (~4 W) — a journey through quantization research, High-Level Synthesis, hardware-software co-design, and system integration.

The result is a working accelerator on AMD's Kria KV260 that restores 64×64×3 satellite image patches in **22.81 ms**, a full **6× faster** than the on-board ARM CPU while consuming only ~4 W of power. The entire pipeline — from PyTorch model to bitstream to live inference via Python — is documented here.

---

## What I Designed

A custom hardware accelerator implementing a Multi-Layer Perceptron (MLP) generator for two satellite image restoration scenarios:

1. **Missing Block Inpainting** — given the top half of a satellite image (32×64×3), predict the bottom half (recovers from sensor dropout). The full reconstructed image is 64×64×3.
2. **Cloud Removal / Denoising** — given a noisy top half (simulating cloud interference), recover the clean version

The generator architecture is a 4-layer MLP:
- **Input**: 6,145 features (32×64×3 pixels flattened + 1 scenario flag)
- **Hidden Layers**: 3 × 512 neurons with LeakyReLU(0.2) activation
- **Output**: 6,144 features with Tanh activation
- **Total Parameters**: ~6.5 million weights

Because the full model (~13 MB at 10-bit) exceeds the KV260's on-chip memory (2.9 MB BRAM+URAM combined), I designed a **DDR streaming architecture** where weights are stored in 4 GB of external DDR4 and burst-loaded to the FPGA fabric on demand. The accelerator uses 10 AXI Master interfaces (one per weight/bias matrix and I/O buffer) multiplexed through an AXI SmartConnect to 4 High Performance ports on the Zynq UltraScale+ MPSoC, maximizing memory bandwidth.

---

## Workflow

### Stage 1: Model Training (PyTorch + Google Colab)

Trained a GAN on the EuroSAT PermanentCrop dataset (~2,500 satellite images) over 300 epochs. The generator and discriminator are jointly optimized with adversarial loss + L1 reconstruction loss. Two scenarios are baked into the dataset: a `flag` value (0.0 or 0.1) at index 0 of the input vector tells the generator which restoration task to perform.

### Stage 2: Quantization Study (Brevitas)

Explored 8-bit, 10-bit, and 12-bit weight quantization to find the sweet spot between accuracy and resource usage. Settled on **10-bit weights with INT16 storage** (scale factor 511), and **FP32 biases** to preserve numerical precision in the accumulators. PSNR drop from FP32 to 10-bit was less than 0.2 dB.

### Stage 3: HLS Implementation (Vitis HLS 2024.2)

Wrote the accelerator in C++ with HLS pragmas:
- `#pragma HLS PIPELINE II=1` on inner MAC loops for one-cycle throughput
- `#pragma HLS INTERFACE m_axi` with `max_read_burst_length=256` and `num_read_outstanding=16` for efficient DDR access
- Bias caching to BRAM at the start of each inference for fast access in the MAC loops
- LUT-based Tanh implementation (avoiding expensive transcendental computation)

C-Simulation verified bit-exact output against a NumPy reference. C-Synthesis reported a 200 MHz clock target with 4 ns achievable latency.

### Stage 4: Block Design (Vivado 2024.2)

Created the system block design integrating:
- Zynq UltraScale+ MPSoC (CPU + DDR controller)
- Custom HLS IP (`network_0`) with 10 m_axi interfaces
- AXI SmartConnect (10 to 4 multiplexer to HP ports)
- AXI Interconnect (for `s_axi_control` register access)
- Processor System Reset

Generating the bitstream pushed my 16 GB laptop to its limits — synthesis ran out of memory on the first attempt. Solved by adding 32 GB of virtual memory and limiting parallel jobs.

### Stage 5: Deployment (PYNQ on Ubuntu KV260)

Booted Ubuntu 22.04 on the KV260, installed PYNQ 3.0.1 via the Kria-PYNQ install script, and accessed JupyterLab at port 9090 via direct Ethernet (192.168.2.1). Loaded the bitstream as a PYNQ Overlay, allocated buffers in DDR using `pynq.allocate()`, set 64-bit physical addresses through the IP's register map, and triggered inference via the `AP_START` control register.

### Stage 6: Verification & Benchmarking

Compared FPGA output against a NumPy reference running the same arithmetic on the on-board ARM Cortex-A53. Measured latency over 100 iterations, throughput, and power consumption via `xmutil xlnx_platformstats --power`.

---

## Execution Results

Image dimension: **64×64×3** (full reconstructed image)

### Performance Comparison (Scenario A: Missing Block Inpainting)

| Metric | CPU (ARM Cortex-A53, FP32) | FPGA (KV260, 10-bit) | Improvement |
|--------|----------------------------|----------------------|-------------|
| Latency (mean ± std) | 137.39 ± 3.12 ms | **22.81 ± 0.01 ms** | **6.02× faster** |
| Throughput | 7.28 fps | **43.84 fps** | **6.02× higher** |
| PSNR vs Ground Truth | 28.51 dB | 28.36 dB | -0.15 dB |
| Numerical accuracy (FPGA vs CPU) | reference | **46.00 dB** | bit-near-exact |

The **46 dB agreement** between FPGA and CPU outputs confirms the hardware implementation is numerically correct — the FPGA reproduces the software reference with negligible error attributable only to 10-bit quantization rounding.

### Performance Comparison (Scenario B: Cloud Removal)

| Metric | CPU (ARM) | FPGA (KV260) |
|--------|-----------|--------------|
| PSNR vs Clean Top | 18.78 dB | 18.78 dB |
| FPGA vs CPU agreement | reference | 43.72 dB |

### Power Consumption

| Mode | Power |
|------|-------|
| Idle | **3.61 W** |
| Load (continuous inference) | **4.01 W** |
| FPGA active overhead | 0.40 W |
| **Power Efficiency** | **10.93 fps/W** |

For comparison, a laptop CPU at 50 W achieves only ~0.3 fps/W on the same model — the KV260 is roughly **35× more power-efficient** than a typical laptop, while also being faster than its own on-board ARM CPU.

### Resource Utilization (Vivado Implementation)

| Resource | Used | Available | Percentage |
|----------|------|-----------|------------|
| BRAM | 39 | 144 | 27% |
| LUT | - | - | 16% |
| DSP | - | 1248 | ~0% |
| FF | - | - | 6% |
| URAM | 0 | 64 | 0% |

The low DSP usage is because the design uses LUT-based MAC operations rather than DSP slices, leaving DSPs available for further accelerators or signal processing. The 27% BRAM usage is largely due to the bias caches and intermediate layer buffers.

### Timing

- Target clock: 200 MHz (5 ns period)
- Achieved (HLS estimate): 4 ns
- Worst Negative Slack: positive (timing met)

### Visual Quality (Scenario A)

The FPGA output is visually indistinguishable from the CPU floating-point reference, with only **0.15 dB PSNR difference** vs ground truth — a difference invisible to the human eye but measurable in objective metrics.

---

## Repository Structure

```
├── 01-training/          # PyTorch GAN training & quantization notebooks
├── 02-hls/               # Vitis HLS C++ source (network.cpp, network.h, etc.)
├── 03-vivado/            # Block design screenshots & resource reports
└── 04-results/           # Visualizations and metric outputs
```

---

## Hardware & Tools

- **Platform**: AMD Kria KV260 Vision AI Starter Kit (Zynq UltraScale+ MPSoC, xck26-sfvc784-2LV-c)
- **Tools**: Vitis HLS 2024.2, Vivado 2024.2, PYNQ 3.0.1
- **OS**: Ubuntu 22.04 LTS on KV260
- **Training**: Python 3.10, PyTorch, Brevitas, Google Colab
- **Dataset**: EuroSAT (PermanentCrop class)

---

## Key Takeaways

This project demonstrates that:

1. **HLS makes FPGA deep learning accessible** — going from a PyTorch model to a working bitstream took ~3 weeks, mostly spent on debugging integration rather than HLS code itself.
2. **Quantization to 10-bit barely affects quality** — 0.15 dB PSNR drop is imperceptible visually, while saving ~3× memory vs FP32.
3. **DDR streaming is necessary for medium-size models** — KV260's 5.1 MB on-chip memory cannot fit a 13 MB model, but external DDR with proper burst configuration achieves usable throughput.
4. **Edge FPGAs deliver real speedup AND power efficiency** — 6× faster than the on-board ARM CPU at less than 1 W active power overhead, achieving 10.93 fps/W. This makes the platform ideal for satellites, drones, and remote sensors where both latency and power are constrained.

---

## Acknowledgments

Developed as part of an internship project at ICDEC. The HLS implementation was inspired by the open-source [gan-hls](https://github.com/dimdano/gan-hls) reference design.

---

