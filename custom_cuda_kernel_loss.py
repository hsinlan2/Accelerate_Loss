import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, Embedding, Sequential, ReLU, Dropout, BatchNorm1d
from torch_geometric.nn import GPSConv, GINEConv
from torch_geometric.data import Data, Batch
from torch import amp
import time
import numpy as np

# Custom CUDA Kernel for Loss Computation 

from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom fused kernel for mutation-aware loss
__global__ void weighted_mutation_loss_kernel(
    const int64_t* __restrict__ input,
    const int64_t* __restrict__ target,
    const float* __restrict__ logits,
    float* __restrict__ loss_output,
    bool* __restrict__ mutation_mask,
    const int n_positions,
    const int n_classes,
    const float mutation_weight
) { //idx for 1D Vector
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_positions) {
        bool is_mutation = input[idx] != target[idx];
        mutation_mask[idx] = is_mutation;
        
        // Initialize max
        int target_class = target[idx];
        float max_logit = logits[idx * n_classes];

        // Find max
        for (int c = 1; c < n_classes; c++) {
            float val = logits[idx * n_classes + c];
            if (val > max_logit) max_logit = val;
        }

        // Compute log-sum-exp
        float logit_sum = 0.0f;
        for (int c = 0; c < n_classes; c++) {
            logit_sum += expf(logits[idx * n_classes + c] - max_logit);
        }

        // Cross entropy loss
        float ce_loss = -logits[idx * n_classes + target_class] + max_logit + logf(logit_sum);

        // Apply mutation weighting
        loss_output[idx] = is_mutation ? ce_loss * mutation_weight : ce_loss;
    }
}

torch::Tensor weighted_mutation_loss_cuda(
    torch::Tensor input,
    torch::Tensor target,
    torch::Tensor logits,
    float mutation_weight
) {
    const int n_positions = input.size(0);
    const int n_classes = logits.size(1);

    auto loss_output = torch::zeros({n_positions}, logits.options());
    auto mutation_mask = torch::zeros({n_positions},
        torch::TensorOptions().dtype(torch::kBool).device(input.device()));

    const int threads = 256;
    const int blocks = (n_positions + threads - 1) / threads;

    weighted_mutation_loss_kernel<<<blocks, threads>>>(
        input.data_ptr<int64_t>(),
        target.data_ptr<int64_t>(),
        logits.data_ptr<float>(),
        loss_output.data_ptr<float>(),
        mutation_mask.data_ptr<bool>(),
        n_positions,
        n_classes,
        mutation_weight
    );

    return loss_output;
}
"""

cpp_source = """
torch::Tensor weighted_mutation_loss_cuda(
    torch::Tensor input, torch::Tensor target,
    torch::Tensor logits, float mutation_weight);
"""

cuda_module = load_inline(
    name='loss_cuda_kernel',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['weighted_mutation_loss_cuda'],
    verbose=True,
    extra_cuda_cflags=['-O3', '--use_fast_math']
)


# Benchmark Function
def benchmark_loss_computation(n_positions=500, n_trials=100):
    device = torch.device('cuda')

    # Create test data
    x_input = torch.randint(0, 21, (n_positions,), device=device)
    y_target = torch.randint(0, 21, (n_positions,), device=device)
    logits = torch.randn(n_positions, 21, device=device)
    mutation_weight = 10.0

    for _ in range(10):
        _ = F.cross_entropy(logits, y_target, reduction='none')

    # PyTorch Run
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_trials):
        mutation_mask = (x_input != y_target)
        aa_loss = F.cross_entropy(logits, y_target, reduction='none')
        weighted_loss = torch.where(mutation_mask, aa_loss * mutation_weight, aa_loss).mean()
    torch.cuda.synchronize()
    time_pytorch = (time.time() - start) / n_trials * 1000

    # CUDA Run
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_trials):
        loss_per_pos = cuda_module.weighted_mutation_loss_cuda(
            x_input, y_target, logits, mutation_weight)
        weighted_loss = loss_per_pos.mean()
    torch.cuda.synchronize()
    time_cuda = (time.time() - start) / n_trials * 1000

    speedup = time_pytorch / time_cuda

    print("\nLoss Computation Benchmark")
    print("Sequence length:", n_positions, "positions")
    print("Number of trials:", n_trials)
    
    print("\nExecution times:")
    print("PyTorch:            ", round(time_pytorch, 3), "ms")
    print("Custom CUDA kernel:  ", round(time_cuda, 3), "ms")
    print("Speedup:             ", round(speedup, 2), "x")
    
    print("\nCustom CUDA kernel achieves", round(speedup, 2), "speedup.")

if __name__ == "__main__":

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}\n")

    benchmark_loss_computation(n_positions=500, n_trials=100)
