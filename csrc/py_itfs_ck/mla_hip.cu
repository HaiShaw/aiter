// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_HIP(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "[HIP ERROR] " << #call << " failed: " << hipGetErrorString(err) << std::endl; \
            return torch::Tensor(); \
        } \
    } while (0)

void print_tensor_shape(const char* name, const torch::Tensor& tensor) {
    auto sizes = tensor.sizes();
    printf("%s shape: [", name);
    for (size_t i = 0; i < sizes.size(); ++i) {
        printf("%ld", sizes[i]);
        if (i != sizes.size() - 1) printf(", ");
    }
    printf("]\n");
}

__global__ void mla_decode_hip_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    int D, int H, int B,
    int S,
    int max_tile
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int idx = b * H + h;

    const float* q_ptr = q + idx * D;
    float* out_ptr = out + idx * D;

    int start = kv_indptr[b];
    int end = kv_indptr[b + 1];
    int len = end - start;

    extern __shared__ float shared_mem[];
    float* scores = shared_mem;
    float* acc = shared_mem + max_tile;
    float* max_buf = shared_mem + max_tile + D;

    for (int d = tid; d < D; d += blockDim.x)
        acc[d] = 0.0f;
    __syncthreads();

    // if (tid == 0 && b == 0 && h == 0) {
    //     printf("acc[0] before tile = %f\n", acc[0]);
    //     printf("len = %d, tile_max = %d\n", len, max_tile);
    // }

    float local_max = -1e9f;
    for (int tile_start = 0; tile_start < len; tile_start += max_tile) {
        int tile_len = min(max_tile, len - tile_start);
        for (int i = tid; i < tile_len; i += blockDim.x) {
            int global_idx = start + tile_start + i;
            int kv_page = kv_indices[global_idx];
            int slot_id = global_idx % S;
            int flat_idx = ((kv_page * S + slot_id) * H + h) * D;
            const float* k_ptr = k + flat_idx;

            float dot = 0.0f;
            for (int d = 0; d < D; ++d)
                dot += q_ptr[d] * k_ptr[d];
            scores[i] = dot;
            local_max = fmaxf(local_max, dot);
        }
        __syncthreads();
    }

    if (tid < blockDim.x)
        max_buf[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            max_buf[tid] = fmaxf(max_buf[tid], max_buf[tid + stride]);
        __syncthreads();
    }
    float e_max = max_buf[0];

    __shared__ float e_sum_shared;
    if (tid == 0) e_sum_shared = 0.0f;
    __syncthreads();

    for (int tile_start = 0; tile_start < len; tile_start += max_tile) {
        int tile_len = min(max_tile, len - tile_start);
        for (int i = tid; i < tile_len; i += blockDim.x) {
            int global_idx = start + tile_start + i;
            int kv_page = kv_indices[global_idx];
            int slot_id = global_idx % S;
            int flat_idx = ((kv_page * S + slot_id) * H + h) * D;
            const float* k_ptr = k + flat_idx;

            float dot = 0.0f;
            for (int d = 0; d < D; ++d)
                dot += q_ptr[d] * k_ptr[d];
            float p = expf(dot - e_max);
            atomicAdd(&e_sum_shared, p);

            const float* v_ptr = v + flat_idx;
            for (int d = 0; d < D; ++d) {
                float weighted = p * v_ptr[d];
                atomicAdd(&acc[d], weighted);
            }
        }
        __syncthreads();
    }

    float e_sum = e_sum_shared;
    for (int d = tid; d < D; d += blockDim.x) {
        out_ptr[d] = acc[d] / (e_sum + 1e-6f);
        // if(d < 8)
        //     printf("out_ptr[%d] = %f, acc[%d] = %f\n", d, out_ptr[d], d, acc[d]);
    }
}

torch::Tensor mla_decode_fwd_hip(torch::Tensor &Q,    //   [batch_size, num_heads, kv_lora_rank + qk_rope_head_dim]
    torch::Tensor &K,                       //   [num_page, page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
    std::optional<torch::Tensor> &v_,        //   [num_page, page_size, num_kv_heads, v_head_dim]
    std::optional<torch::Tensor> &out_,        //   [batch_size, num_heads, v_head_dim]
    int head_size_v,
    torch::Tensor &kv_indptr,               //   [batch_size+1]
    torch::Tensor &kv_page_indices,         //   [num_page_used]
    torch::Tensor &kv_last_page_lens,       //   [batch_size]
    float softmax_scale)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(Q));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor Qf = Q.to(torch::kFloat32);
    torch::Tensor Kf = K.to(torch::kFloat32);
    torch::Tensor V = v_.value_or(K).slice(-1, 0, head_size_v).to(torch::kFloat32);
    torch::Tensor O = out_.value_or(torch::empty_like(Qf)).to(torch::kFloat32);

    // printf("===================================\n");
    // print_tensor_shape("Q", Q);
    // print_tensor_shape("K", K);
    // print_tensor_shape("V", V);
    // print_tensor_shape("O", O);
    // print_tensor_shape("kv_indptr", kv_indptr);
    // print_tensor_shape("kv_page_indices", kv_page_indices);
    // print_tensor_shape("kv_last_page_lens", kv_last_page_lens);

    const float* q_ptr = Qf.data_ptr<float>();
    const float* k_ptr = Kf.data_ptr<float>();
    const float* v_ptr = V.data_ptr<float>();
    float* out_ptr = O.data_ptr<float>();

    const int* indptr_ptr = kv_indptr.data_ptr<int>();
    const int* indices_ptr = kv_page_indices.data_ptr<int>();

    int B = Qf.size(0);
    int H = Qf.size(1);
    int D = Qf.size(2);
    int P = Kf.size(0);
    int S = Kf.size(1);

    int tile_max = std::min(P * S, 1024);
    size_t shared_mem = (tile_max + D + tile_max) * sizeof(float);

    dim3 grid(B, H);
    dim3 block(tile_max);

    // hipDeviceProp_t props;
    // hipGetDeviceProperties(&props, 0);  // device 0
    // std::cout << "Device name: " << props.name << std::endl;
    // std::cout << "Max threads per block: " << props.maxThreadsPerBlock << std::endl;
    // std::cout << "Shared memory per block: " << props.sharedMemPerBlock << " bytes" << std::endl;
    // std::cout << "Max block dim (x): " << props.maxThreadsDim[0] << std::endl;
    // std::cout << "Max grid dim (x): " << props.maxGridSize[0] << std::endl;


    // std::cout << "Launching kernel with: B=" << B << ", H=" << H << ", D=" << D
    // << ", P=" << P << ", S=" << S << ", blockDim.x=" << std::max(P*S, D)
    // << ", shared=" << shared_mem << " bytes\n";

    mla_decode_hip_kernel<<<grid, block, shared_mem, stream>>>(
        q_ptr, k_ptr, v_ptr, out_ptr,
        indptr_ptr, indices_ptr,
        D, H, B, S, tile_max);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    auto O_cpu = O.cpu();
    auto accessor = O_cpu.accessor<float, 3>();

    // for (int i = 0; i < 8; ++i)
    //     std::cout << "[host check] O[0][0][" << i << "] = " << accessor[0][0][i] << std::endl;

    return O.to(torch::kBFloat16);
}