# TinyLLM — C++/CUDA Transformer From Scratch

**Status:** Design specification v0.1  
**Owner:** Project developer  
**Target:** Linux, C++20, NVIDIA CUDA, single consumer NVIDIA GPU (initial target: RTX 4060 laptop, ~8 GB VRAM)  
**Build system:** GNU Make, with explicit C++ and CUDA compilation/linking  
**Implementation policy:** The developer writes all code. An AI collaborator should give specifications, conceptual explanations, interface reviews, and tests; **do not write code unless explicitly requested**.

## 1. Goal and non-goals

**Primary goal:** Build and train a small, decoder-only autoregressive transformer from randomly initialized weights using code authored in C++ and CUDA. Produce a reproducible CPU reference implementation, a functionally equivalent CUDA implementation, and a working text-generation executable.

**First success criterion:** Given a tiny corpus, training loss decreases and the model can overfit one small batch; given a starting prompt, an inference loop generates tokens autoregressively. Track validation loss separately so memorization is not mistaken for generalization.

**Out of scope for v0.1:** ChatGPT-quality conversations, distributed training, production serving, FlashAttention, elaborate tokenizers, quantization, multimodal input, RLHF, and GPU kernel optimization before correctness.

**Key philosophy:** correctness → end-to-end learning → CPU/GPU parity → profiling/optimization.

## 2. High-level system

Training data (UTF-8 bytes) → byte tokenizer → sequence batches (input/shifted target) → embeddings → repeated decoder blocks → final normalization → vocabulary projection → logits → cross-entropy → backward pass → AdamW update → checkpoint.

Generation: prompt → tokenizer → model forward → last-position logits → sampling → append selected byte → repeat → decode.

### Decoder block (pre-normalization)

Input X → layer normalization → multi-head masked self-attention → residual addition → layer normalization → feed-forward network (2 linear layers with GELU) → residual addition → output.

For attention: Q=XWq, K=XWk, V=XWv; per-head output = softmax((QKᵀ/√d_head) + causal_mask)V. Project concatenated heads back to model width. Mask inaccessible future positions **before** softmax. Include biases consistently or deliberately omit them and document the choice.

## 3. Initial model configuration

These are starting defaults, not fixed interface assumptions:

| Parameter | Default |
|---|---:|
| Tokenization | byte-level, vocabulary 256 |
| Context length T | 128 |
| Embedding width C | 128 |
| Decoder blocks L | 4 |
| Attention heads H | 4 |
| Head width C/H | 32 |
| MLP hidden width | 512 (4C) |
| Train batch size B | 4 initially |
| Data type | FP32 initially |
| Initialization | seeded random, scaled appropriately |
| Optimizer | AdamW |

This configuration has approximately 0.86 million parameters with tied input/output embeddings and standard GPT-style linear layers. Expand toward ~2M only after the small model trains reliably. The model predicts **next bytes** (not necessarily human words). It will be a tiny language model, not a chat assistant.

## 4. Code organization

```text
tiny-llm/
├── Makefile
├── README.md
├── docs/
│   └── DESIGN.md
├── include/
│   ├── tensor.hpp
│   ├── ops.hpp
│   ├── linear.hpp
│   ├── embedding.hpp
│   ├── attention.hpp
│   ├── transformer.hpp
│   ├── loss.hpp
│   ├── optimizer.hpp
│   ├── tokenizer.hpp
│   ├── data_loader.hpp
│   └── checkpoint.hpp
├── src/
│   ├── tensor.cpp
│   ├── ops.cpp
│   ├── linear.cpp
│   ├── embedding.cpp
│   ├── attention.cpp
│   ├── transformer.cpp
│   ├── loss.cpp
│   ├── optimizer.cpp
│   ├── tokenizer.cpp
│   ├── data_loader.cpp
│   ├── checkpoint.cpp
│   ├── train.cpp
│   └── generate.cpp
├── cuda/
│   ├── matmul.cuh / matmul.cu
│   ├── softmax.cuh / softmax.cu
│   ├── layernorm.cuh / layernorm.cu
│   ├── attention.cuh / attention.cu
│   └── device_buffer.cuh / device_buffer.cu
├── tests/
│   ├── tensor_tests.cpp
│   ├── ops_tests.cpp
│   ├── grad_tests.cpp
│   ├── model_tests.cpp
│   └── cpu_gpu_parity_tests.cpp
├── data/                  # local training input; avoid committing corpora
├── checkpoints/           # outputs; do not commit
└── build/                 # generated; do not commit
```

File list is a target architecture; create files incrementally. `.hpp` defines ordinary C++ APIs; `.cpp` implements CPU/model behavior; `.cuh` declares GPU-facing interfaces, device helpers, and templates; `.cu` implements CUDA routines. A separate `.cuh`/`.cu` does **not** intrinsically slow execution. Cross-translation-unit inlining and link-time optimization can matter later, but kernel count, launch overhead, memory movement, and algorithms usually matter more.

## 5. Core design decisions

1. **Reference first:** CPU implementation is authoritative for numerical correctness. CUDA must match within a documented floating-point tolerance (not bit-identical).
2. **Row-major tensors:** Values stored contiguously; logical shapes and strides maintained explicitly. Detect size overflow, invalid indices, and incompatible shapes.
3. **Ownership:** CPU `Tensor` owns host memory; later `DeviceBuffer` or `DeviceTensor` owns GPU memory via RAII. No implicit host↔device transfers on ordinary element access.
4. **Tensor dtype:** FP32 for initial forward and backward passes. Token IDs are integer data and should be handled separately from float tensors.
5. **Batch semantics:** Sequence training uses input bytes `data[i ... i+T-1]` and shifted targets `data[i+1 ... i+T]`. Keep train/validation splits disjoint.
6. **Model components:** Distinguish **parameters**, **activations**, and **gradients**. Every trainable parameter must have a gradient buffer and optimizer state.
7. **Backprop:** Manual backward passes are preferred for learning, with finite-difference gradient checking for small shapes. Avoid assuming that forward-only code is trainable.
8. **Stable numerics:** Softmax subtracts row maximum; cross-entropy should use log-sum-exp or an equivalent stable expression. Layer norm includes epsilon.
9. **Randomness:** Seed model initialization, data sampling, and token sampling; save seed/config in checkpoints.
10. **Checkpoints:** Versioned format with model configuration and parameter tensors; later include optimizer state and step to allow actual training resumption.

## 6. Component contracts

| Component | Input | Output | Acceptance criteria |
|---|---|---|---|
| CPU Tensor | shape and FP32 values | allocated contiguous storage and indexed access | correct strides, bounds checks, allocation and copy/move semantics |
| Matrix multiplication | A [M,K], B [K,N] | C [M,N] | matches hand-computed cases; rejects invalid dimensions |
| Broadcast/add & elementwise ops | shape-compatible tensors | elementwise outputs | dimension and edge-case tests |
| Linear | X [...,Cin], W [Cin,Cout], bias [Cout] | Y [...,Cout] | matches manually computed result |
| Embedding | token IDs [B,T] | X [B,T,C] | exact row lookup, token range checking |
| LayerNorm | X [...,C] | normalized X of same shape | mean/variance calculations verified including constant vector |
| GELU | X | activated X | approximate/reference values and gradients |
| Attention | X [B,T,C] | X' [B,T,C] | heads reshape correctly; future tokens cannot affect earlier outputs |
| Feed-forward | X [B,T,C] | Y [B,T,C] | width C→4C→C with activation |
| Transformer | token IDs [B,T] | logits [B,T,V] | shape and masking invariant preserved through L blocks |
| Cross-entropy | logits [B,T,V], targets [B,T] | scalar loss plus logits gradient | uniform logits loss ≈ ln(V); gradient check |
| Backward passes | cached forward activations + upstream gradients | input/weight/bias gradients | finite differences agree on small random tensors |
| AdamW | parameter, gradient, moments, step | updated parameter/moments | one-step arithmetic checked; decoupled weight decay |
| Tokenizer | bytes | IDs [0,255] | lossless byte round-trip |
| Data loader | token array, B,T | input and shifted target batches | no out-of-range access; exact one-position shift |
| Generator | prompt, model, decoding settings | sampled continuation | single-step and repeated-step outputs well-formed |
| Checkpoint | model + metadata | saved/loaded model | round-trip yields same logits within tolerance |

## 7. Training and inference invariants

- Training forward produces logits for **every** sequence position, not just the last one.
- Causal masking ensures a change to token `t+1` cannot affect logits at position `t`.
- Training backward accumulates gradients correctly; gradient buffers reset or accumulate intentionally between updates.
- Loss on a single tiny batch should fall substantially and approach a very low value under an appropriate learning rate before scaling data/model.
- Validation is measured on held-out text; training loss alone cannot establish language ability.
- In generation, use only logits at the final position, sample or argmax a byte, append, repeat; crop context to T if necessary.
- When reading generated bytes, arbitrary byte sequences may not form valid UTF-8; handle decoding/display safely.

## 8. GNU Make build contract

**Decision:** Use GNU Make initially because the project is small, Linux-focused, and deliberately low-level. This is not inherently faster than CMake: CMake generates builds; the runtime is determined by the resulting binaries and compiler options. Reconsider CMake if dependencies, portability, or multiple platforms become cumbersome.

Makefile should eventually provide these targets:

| Target | Required behavior |
|---|---|
| `make` or `make all` | compile main CPU training/generation executables |
| `make test` | build and run CPU unit tests |
| `make cuda` | build CUDA-enabled targets and required GPU objects |
| `make test-cuda` | build/run CPU↔GPU parity tests (requires GPU) |
| `make debug` | use debug symbols, warnings, suitable debugging configuration |
| `make release` | optimized build with reproducible flags |
| `make clean` | remove generated build artifacts only |
| `make help` | describe targets and relevant variables |

Build constraints:

- Compile `.cpp` with the C++ compiler and `.cu` with `nvcc`; use an explicit rule for each, with compiler-generated header dependencies where feasible.
- Set C++ standard consistently (initially C++20 if supported by local NVCC/toolchain; otherwise use C++17 consistently).
- Use dedicated `build/` object folders; do not put generated files beside source files.
- Set the NVIDIA GPU architecture intentionally for the target machine; make this overridable in a Make variable.
- When linking CUDA object code into a C++ executable, arrange CUDA runtime linkage and any required CUDA device-link step. Using NVCC for final linking of CUDA-enabled targets is simplest initially.
- Device code in separate `.cu` translation units may need relocatable device code/device linking if device symbols are called across units; **avoid enabling it unless the design needs it**.
- Only rebuild changed dependencies. Preserve compiler/linker errors rather than suppressing them.
- Expose `CXX`, `NVCC`, `CUDA_ARCH`, build flags, and output directory as configurable variables.
- Debug mode is for checking correctness; optimize only after CPU/GPU parity is established.

No implementation code is specified here. The developer writes the Makefile.

## 9. Milestones and gates

| Phase | Deliverable | Exit test |
|---|---|---|
| M0: build scaffold | directory layout, Makefile, runnable tests | clean build and repeatable `make test` |
| M1: tensor | CPU FP32 row-major tensor | shape/stride/index/ownership edge cases pass |
| M2: math ops | matmul, elementwise, softmax, normalization | numerical reference tests pass |
| M3: neural primitives | embedding, linear, GELU, normalization | outputs/shapes match references |
| M4: transformer forward | masked multi-head attention and blocks | causality and dimensions tests pass |
| M5: training | manual backward, loss, AdamW, dataloader | gradient checks pass; overfit one batch |
| M6: usable tiny LM | full dataset training, validation, checkpoints, generation | sampled continuations; checkpoint reproducibility |
| M7: CUDA port | GPU storage and forward/backward ops | CPU/GPU parity on small deterministic cases |
| M8: optimization | profiled CUDA kernels and benchmarks | measurable latency/throughput improvements without accuracy regression |

## 10. CUDA performance measurements (later, not now)

Record tokens/sec for training and generation, peak GPU memory, per-kernel latency, transfer volume/time, achieved occupancy where meaningful, and GPU utilization. Compare naive GEMM against tiled GEMM; measure attention's T² scaling. Profile before selecting an optimization. Separate correctness tests from performance benchmarks.

## 11. Optional security extension (future)

Once save/load is stable, define a signed or digest-verified manifest for checkpoints and per-layer parameter hashes. Verify these on model load and optionally before inference sessions. This protects against unverified checkpoint modifications **at verification boundaries**; it does not guarantee safety against arbitrary runtime memory corruption. Treat this as a later phase, not a prerequisite for the first training run.

## 12. Collaboration instructions for Cowork / coding assistant

**Working agreement:**

- The developer wants to write code independently. **Do not generate functions, full files, patches, or Makefile recipes unless explicitly requested.**
- For each task, supply: goal, required math, shape contract, memory ownership constraints, edge cases, tests, and definition of done.
- Review submitted code by pointing out concrete defects, explaining why they matter, and suggesting fixes conceptually before offering code.
- Work on one milestone at a time; do not scaffold every source file prematurely.
- When discussing GPU optimization, distinguish launch overhead, device memory accesses, host-device transfers, and compiler/linking concerns.
- Favor transparent implementation over external ML frameworks. Standard C++ library and CUDA runtime APIs are allowed. Do not rely on PyTorch/TensorFlow to implement model behavior.
- Any changes to scope or numerical conventions should be recorded in this document.

**Immediate next task (M0):** Create the directory scaffold and author a Makefile supporting CPU tests and a later CUDA target. Then implement the CPU tensor with contiguous row-major FP32 storage. No need to write attention or CUDA kernels yet.

## 13. Decisions to revisit

- Manual gradients versus building a small autodiff engine: manual initially.
- Biases in all linear layers versus selected layers: select and document before M4.
- Tied embeddings versus separate output matrix: tied initially.
- Positional representation: learned absolute positional embeddings initially.
- Norm variant: LayerNorm initially.
- GELU exact versus tanh approximation: choose one and use it consistently in reference and CUDA versions.
- Checkpoint format: simple versioned binary with explicit tensor names/shapes.
