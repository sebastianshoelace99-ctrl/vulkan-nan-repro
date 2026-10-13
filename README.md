# burn 0.22 / cubecl 0.11 Vulkan NaN repro (Aeternus MoE, 2-block truncation)

`cargo run --release`

Loads real trained weights (data/*.npy, exported from a PyTorch checkpoint),
runs the forward pass on the Vulkan backend, and prints:
1. the router-logits materialization (expected: matches PyTorch; observed: all-NaN)
2. the full model logits vs PyTorch (observed: bit-exact, max abs diff 0.00000)
3. a CrossEntropy probe on fresh tensors (observed: correct) vs on the model
   logits (observed: NaN) - demonstrating the multi-consumer/re-execution bug.

 burn = { version = "0.22.0", features = ["vulkan", "fusion"] }
