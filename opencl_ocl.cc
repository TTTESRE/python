#include <torch/extension.h>

#include <CL/cl.h>

#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <utility>

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>

namespace {

std::mutex ocl_init_mutex;
cl_context ocr_ctx = nullptr;
cl_command_queue ocr_queue = nullptr;
cl_device_id ocr_device = nullptr;
cl_kernel ocr_k_linear = nullptr;
cl_kernel ocr_k_linear_relu = nullptr;
cl_kernel ocr_k_linear_tanh = nullptr;
cl_kernel ocr_k_relu = nullptr;
cl_kernel ocr_k_tanh = nullptr;
cl_kernel ocr_k_relu_bwd = nullptr;
cl_kernel ocr_k_tanh_bwd = nullptr;
cl_kernel ocr_k_linear_bwd = nullptr;
bool ocr_initialized = false;

std::unordered_map<void*, std::pair<cl_mem, size_t>> g_param_cache;

const char* CL_SOURCE_SGEMM_HEAD = R"(
__kernel void sgemm(__global const float* W,
                    __global const float* X,
                    __global const float* B,
                    __global float* Y,
                    int M, int K, int N) {
    const int TILE = 16;
    __local float As[TILE*TILE];
    __local float Bs[TILE*TILE];
    int row = get_global_id(0);
    int col = get_global_id(1);
    float sum = 0.0f;
    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int kw = t*TILE + get_local_id(0);
        As[get_local_id(1)*TILE + get_local_id(0)] = (col < N && kw < K) ? W[col*K + kw] : 0.0f;
        int kx = t*TILE + get_local_id(1);
        Bs[get_local_id(0)*TILE + get_local_id(1)] = (row < M && kx < K) ? X[row*K + kx] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < TILE; ++i)
            sum += As[get_local_id(1)*TILE + i] * Bs[get_local_id(0)*TILE + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N)
        Y[row*N + col] = sum + B[col];
)";

const char* CL_SOURCE_SGEMM_TAIL = R"(
}
)";

static std::string sgemm_src(const char* activation_line) {
    return std::string(CL_SOURCE_SGEMM_HEAD) + "\n" + activation_line + "\n" + std::string(CL_SOURCE_SGEMM_TAIL);
}

const std::string SRC_LINEAR      = sgemm_src("// no activation");
const std::string SRC_LINEAR_RELU = sgemm_src("if (row < M && col < N) Y[row*N + col] = (Y[row*N + col] > 0.0f) ? Y[row*N + col] : 0.0f;");
const std::string SRC_LINEAR_TANH = sgemm_src("if (row < M && col < N) Y[row*N + col] = tanh(Y[row*N + col]);");

const char* CL_SOURCE_SGEMM_STD = R"(
__kernel void sgemm_std(int M, int N, int K,
                        __global const float* A,
                        __global const float* B,
                        __global const float* bias,
                        __global float* C) {
    const int TILE = 16;
    __local float a_tile[TILE*TILE];
    __local float b_tile[TILE*TILE];
    int row = get_global_id(0);
    int col = get_global_id(1);
    float sum = 0.0f;
    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int ka = t*TILE + get_local_id(0);
        a_tile[get_local_id(1)*TILE + get_local_id(0)] = (row < M && ka < K) ? A[row*K + ka] : 0.0f;
        int kb = t*TILE + get_local_id(1);
        b_tile[get_local_id(0)*TILE + get_local_id(1)] = (col < N && kb < K) ? B[kb*N + col] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < TILE; ++i)
            sum += a_tile[get_local_id(1)*TILE + i] * b_tile[get_local_id(0)*TILE + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N)
        C[row*N + col] = sum + bias[col];
}
)";

const char* CL_SOURCE_RELU = R"(
__kernel void relu_forward(__global const float* X, __global float* Y, int n) {
    int i = get_global_id(0);
    if (i < n) Y[i] = fmax(X[i], 0.0f);
}
)";

const char* CL_SOURCE_TANH = R"(
__kernel void tanh_forward(__global const float* X, __global float* Y, int n) {
    int i = get_global_id(0);
    if (i < n) Y[i] = tanh(X[i]);
}
)";

const char* CL_SOURCE_RELU_BWD = R"(
__kernel void relu_backward(__global const float* G, __global const float* X, __global float* Y, int n) {
    int i = get_global_id(0);
    if (i < n) Y[i] = X[i] > 0.0f ? G[i] : 0.0f;
}
)";

const char* CL_SOURCE_TANH_BWD = R"(
__kernel void tanh_backward(__global const float* G, __global const float* Y, __global float* O, int n) {
    int i = get_global_id(0);
    if (i < n) O[i] = G[i] * (1.0f - Y[i]*Y[i]);
}
)";

std::string ocl_error_str(int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        default: return "CL_UNKNOWN(" + std::to_string(err) + ")";
    }
}

void ensure_init() {
    if (ocr_initialized) return;
    std::lock_guard<std::mutex> lock(ocl_init_mutex);
    if (ocr_initialized) return;

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found, using CPU fallback" << std::endl;
        ocr_initialized = true;
        return;
    }

    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(),
        0
    };

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);
    if (devices.empty()) {
        std::cerr << "No OpenCL devices found, using CPU fallback" << std::endl;
        ocr_initialized = true;
        return;
    }

    ocr_device = devices[0]();
    std::cout << "OpenCL device: " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;

    ocr_ctx = clCreateContext(props, 1, &ocr_device, nullptr, nullptr, nullptr);
    if (!ocr_ctx) {
        std::cerr << "Failed to create OpenCL context, using CPU fallback" << std::endl;
        ocr_initialized = true;
        return;
    }

    ocr_queue = clCreateCommandQueue(ocr_ctx, ocr_device, 0, nullptr);
    if (!ocr_queue) {
        clReleaseContext(ocr_ctx);
        ocr_ctx = nullptr;
        std::cerr << "Failed to create OpenCL command queue, using CPU fallback" << std::endl;
        ocr_initialized = true;
        return;
    }

    auto build_prog = [&](const std::string& src, cl_kernel* k, const char* name) {
        const char* ptr = src.c_str();
        cl_program prog = clCreateProgramWithSource(ocr_ctx, 1, &ptr, nullptr, nullptr);
        if (!prog) return;
        cl_int err = clBuildProgram(prog, 1, &ocr_device, "", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            char log[4096];
            clGetProgramBuildInfo(prog, ocr_device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
            std::cerr << "OpenCL build log for " << name << ": " << log << std::endl;
            clReleaseProgram(prog);
            return;
        }
        cl_int err2;
        *k = clCreateKernel(prog, name, &err2);
        if (!*k) std::cerr << "OpenCL kernel create failed for " << name << " err=" << err2 << std::endl;
        clReleaseProgram(prog);
    };

    build_prog(SRC_LINEAR, &ocr_k_linear, "sgemm");
    build_prog(SRC_LINEAR_RELU, &ocr_k_linear_relu, "sgemm");
    build_prog(SRC_LINEAR_TANH, &ocr_k_linear_tanh, "sgemm");
    build_prog(CL_SOURCE_RELU, &ocr_k_relu, "relu_forward");
    build_prog(CL_SOURCE_TANH, &ocr_k_tanh, "tanh_forward");
    build_prog(CL_SOURCE_RELU_BWD, &ocr_k_relu_bwd, "relu_backward");
    build_prog(CL_SOURCE_TANH_BWD, &ocr_k_tanh_bwd, "tanh_backward");
    build_prog(CL_SOURCE_SGEMM_STD, &ocr_k_linear_bwd, "sgemm_std");

    ocr_initialized = true;
}

inline size_t round_up(size_t v, size_t m) { return (v + m - 1) / m * m; }

cl_mem get_param_buffer(const float* host, size_t nbytes) {
    void* key = const_cast<float*>(host);
    auto it = g_param_cache.find(key);
    if (it != g_param_cache.end() && it->second.second == nbytes) {
        clEnqueueWriteBuffer(ocr_queue, it->second.first, CL_TRUE, 0, nbytes, host, 0, nullptr, nullptr);
        return it->second.first;
    }
    cl_mem buf = clCreateBuffer(ocr_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nbytes, const_cast<float*>(host), nullptr);
    if (!buf) throw std::runtime_error("Failed to allocate OpenCL param buffer");
    g_param_cache[key] = {buf, nbytes};
    return buf;
}

cl_mem alloc_buffer(size_t nbytes) {
    cl_mem buf = clCreateBuffer(ocr_ctx, CL_MEM_READ_WRITE, nbytes, nullptr, nullptr);
    if (!buf) throw std::runtime_error("Failed to allocate OpenCL buffer");
    return buf;
}

void run_linear_kernel(cl_kernel k, cl_mem dW, cl_mem dX, cl_mem dB, cl_mem dY, int M, int N, int K) {
    clSetKernelArg(k, 0, sizeof(cl_mem), &dW);
    clSetKernelArg(k, 1, sizeof(cl_mem), &dX);
    clSetKernelArg(k, 2, sizeof(cl_mem), &dB);
    clSetKernelArg(k, 3, sizeof(cl_mem), &dY);
    clSetKernelArg(k, 4, sizeof(int), &M);
    clSetKernelArg(k, 5, sizeof(int), &K);
    clSetKernelArg(k, 6, sizeof(int), &N);
    size_t gws[2] = {round_up((size_t)M, 16), round_up((size_t)N, 16)};
    size_t lws[2] = {16, 16};
    cl_int err = clEnqueueNDRangeKernel(ocr_queue, k, 2, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        throw std::runtime_error("clEnqueueNDRangeKernel failed: " + ocl_error_str(err));
}

void run_std_kernel(cl_kernel k, cl_mem dA, cl_mem dB, cl_mem dBias, cl_mem dC, int M, int N, int K) {
    clSetKernelArg(k, 0, sizeof(int), &M);
    clSetKernelArg(k, 1, sizeof(int), &N);
    clSetKernelArg(k, 2, sizeof(int), &K);
    clSetKernelArg(k, 3, sizeof(cl_mem), &dA);
    clSetKernelArg(k, 4, sizeof(cl_mem), &dB);
    clSetKernelArg(k, 5, sizeof(cl_mem), &dBias);
    clSetKernelArg(k, 6, sizeof(cl_mem), &dC);
    size_t gws[2] = {round_up((size_t)M, 16), round_up((size_t)N, 16)};
    size_t lws[2] = {16, 16};
    cl_int err = clEnqueueNDRangeKernel(ocr_queue, k, 2, nullptr, gws, lws, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        throw std::runtime_error("clEnqueueNDRangeKernel failed: " + ocl_error_str(err));
}

void run_unary_kernel(cl_kernel k, cl_mem dX, cl_mem dY, int n) {
    clSetKernelArg(k, 0, sizeof(cl_mem), &dX);
    clSetKernelArg(k, 1, sizeof(cl_mem), &dY);
    int ni = n;
    clSetKernelArg(k, 2, sizeof(int), &ni);
    size_t gws = round_up((size_t)n, 64);
    cl_int err = clEnqueueNDRangeKernel(ocr_queue, k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        throw std::runtime_error("clEnqueueNDRangeKernel failed: " + ocl_error_str(err));
}

void run_binary_unary_kernel(cl_kernel k, cl_mem dX, cl_mem dY, cl_mem dO, int n) {
    clSetKernelArg(k, 0, sizeof(cl_mem), &dX);
    clSetKernelArg(k, 1, sizeof(cl_mem), &dY);
    clSetKernelArg(k, 2, sizeof(cl_mem), &dO);
    int ni = n;
    clSetKernelArg(k, 3, sizeof(int), &ni);
    size_t gws = round_up((size_t)n, 64);
    cl_int err = clEnqueueNDRangeKernel(ocr_queue, k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        throw std::runtime_error("clEnqueueNDRangeKernel failed: " + ocl_error_str(err));
}

torch::Tensor linear_kernel_dispatch(cl_kernel k, torch::Tensor weight, torch::Tensor bias, torch::Tensor input) {
    ensure_init();
    auto W = weight.contiguous().to(torch::kFloat32);
    auto B = bias.contiguous().to(torch::kFloat32);
    auto X = input.contiguous().to(torch::kFloat32);

    int64_t M = X.size(0);
    int64_t K = W.size(1);
    int64_t N = W.size(0);

    if (!ocr_initialized || !k || !ocr_queue) {
        return X.matmul(W.t()).add(B);
    }

    auto out = torch::empty({M, N}, torch::kFloat32);
    const float* wp = W.data_ptr<float>();
    const float* bp = B.data_ptr<float>();
    const float* xp = X.data_ptr<float>();
    float* op = out.data_ptr<float>();

    cl_mem dW = get_param_buffer(wp, (size_t)N * K * sizeof(float));
    cl_mem dB = get_param_buffer(bp, (size_t)N * sizeof(float));
    cl_mem dX = alloc_buffer((size_t)M * K * sizeof(float));
    cl_mem dY = alloc_buffer((size_t)M * N * sizeof(float));

    clEnqueueWriteBuffer(ocr_queue, dX, CL_TRUE, 0, (size_t)M * K * sizeof(float), xp, 0, nullptr, nullptr);
    run_linear_kernel(k, dW, dX, dB, dY, (int)M, (int)N, (int)K);
    clFinish(ocr_queue);
    clEnqueueReadBuffer(ocr_queue, dY, CL_TRUE, 0, (size_t)M * N * sizeof(float), op, 0, nullptr, nullptr);

    clReleaseMemObject(dX);
    clReleaseMemObject(dY);
    return out;
}

torch::Tensor std_kernel_dispatch(cl_kernel k, torch::Tensor A, torch::Tensor B, torch::Tensor bias) {
    ensure_init();
    auto a = A.contiguous().to(torch::kFloat32);
    auto b = B.contiguous().to(torch::kFloat32);
    auto bias_t = bias.contiguous().to(torch::kFloat32);

    int64_t M = a.size(0);
    int64_t K = a.size(1);
    int64_t N = b.size(1);

    if (!ocr_initialized || !k || !ocr_queue) {
        return a.matmul(b).add(bias_t);
    }

    auto out = torch::empty({M, N}, torch::kFloat32);
    const float* ap = a.data_ptr<float>();
    const float* bp = b.data_ptr<float>();
    const float* biasp = bias_t.data_ptr<float>();
    float* op = out.data_ptr<float>();

    cl_mem dA = alloc_buffer((size_t)M * K * sizeof(float));
    cl_mem dB = alloc_buffer((size_t)K * N * sizeof(float));
    cl_mem dBias = alloc_buffer((size_t)N * sizeof(float));
    cl_mem dC = alloc_buffer((size_t)M * N * sizeof(float));

    clEnqueueWriteBuffer(ocr_queue, dA, CL_TRUE, 0, (size_t)M * K * sizeof(float), ap, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(ocr_queue, dB, CL_TRUE, 0, (size_t)K * N * sizeof(float), bp, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(ocr_queue, dBias, CL_TRUE, 0, (size_t)N * sizeof(float), biasp, 0, nullptr, nullptr);
    run_std_kernel(k, dA, dB, dBias, dC, (int)M, (int)N, (int)K);
    clFinish(ocr_queue);
    clEnqueueReadBuffer(ocr_queue, dC, CL_TRUE, 0, (size_t)M * N * sizeof(float), op, 0, nullptr, nullptr);

    clReleaseMemObject(dA);
    clReleaseMemObject(dB);
    clReleaseMemObject(dBias);
    clReleaseMemObject(dC);
    return out;
}

torch::Tensor unary_kernel_dispatch(cl_kernel k, torch::Tensor input) {
    ensure_init();
    auto X = input.contiguous().to(torch::kFloat32);
    int64_t n = X.numel();

    if (!ocr_initialized || !k || !ocr_queue) {
        auto out = torch::empty_like(X);
        float* op = out.data_ptr<float>();
        const float* xp = X.data_ptr<float>();
        for (int64_t i = 0; i < n; ++i) op[i] = (k == ocr_k_relu) ? std::max(xp[i], 0.0f) : std::tanh(xp[i]);
        return out;
    }

    auto out = torch::empty_like(X);
    const float* xp = X.data_ptr<float>();
    float* op = out.data_ptr<float>();
    cl_mem dX = alloc_buffer((size_t)n * sizeof(float));
    cl_mem dY = alloc_buffer((size_t)n * sizeof(float));
    clEnqueueWriteBuffer(ocr_queue, dX, CL_TRUE, 0, (size_t)n * sizeof(float), xp, 0, nullptr, nullptr);
    run_unary_kernel(k, dX, dY, (int)n);
    clFinish(ocr_queue);
    clEnqueueReadBuffer(ocr_queue, dY, CL_TRUE, 0, (size_t)n * sizeof(float), op, 0, nullptr, nullptr);
    clReleaseMemObject(dX);
    clReleaseMemObject(dY);
    return out;
}

void cleanup_ocl() {
    if (!ocr_initialized) return;
    for (auto& kv : g_param_cache) clReleaseMemObject(kv.second.first);
    g_param_cache.clear();
    if (ocr_k_linear) clReleaseKernel(ocr_k_linear);
    if (ocr_k_linear_relu) clReleaseKernel(ocr_k_linear_relu);
    if (ocr_k_linear_tanh) clReleaseKernel(ocr_k_linear_tanh);
    if (ocr_k_relu) clReleaseKernel(ocr_k_relu);
    if (ocr_k_tanh) clReleaseKernel(ocr_k_tanh);
    if (ocr_k_relu_bwd) clReleaseKernel(ocr_k_relu_bwd);
    if (ocr_k_tanh_bwd) clReleaseKernel(ocr_k_tanh_bwd);
    if (ocr_k_linear_bwd) clReleaseKernel(ocr_k_linear_bwd);
    if (ocr_queue) clReleaseCommandQueue(ocr_queue);
    if (ocr_ctx) clReleaseContext(ocr_ctx);
    ocr_k_linear = nullptr;
    ocr_k_linear_relu = nullptr;
    ocr_k_linear_tanh = nullptr;
    ocr_k_relu = nullptr;
    ocr_k_tanh = nullptr;
    ocr_k_relu_bwd = nullptr;
    ocr_k_tanh_bwd = nullptr;
    ocr_k_linear_bwd = nullptr;
    ocr_queue = nullptr;
    ocr_ctx = nullptr;
    ocr_initialized = false;
}

} // namespace

torch::Tensor linear_forward(torch::Tensor weight, torch::Tensor bias, torch::Tensor input) {
    return linear_kernel_dispatch(ocr_k_linear, weight, bias, input);
}

torch::Tensor linear_relu_forward(torch::Tensor weight, torch::Tensor bias, torch::Tensor input) {
    return linear_kernel_dispatch(ocr_k_linear_relu, weight, bias, input);
}

torch::Tensor linear_tanh_forward(torch::Tensor weight, torch::Tensor bias, torch::Tensor input) {
    return linear_kernel_dispatch(ocr_k_linear_tanh, weight, bias, input);
}

torch::Tensor relu_forward(torch::Tensor input) {
    return unary_kernel_dispatch(ocr_k_relu, input);
}

torch::Tensor tanh_forward(torch::Tensor input) {
    return unary_kernel_dispatch(ocr_k_tanh, input);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> linear_backward(
    torch::Tensor grad_output, torch::Tensor weight, torch::Tensor input) {
    ensure_init();
    auto go = grad_output.contiguous().to(torch::kFloat32);
    auto W = weight.contiguous().to(torch::kFloat32);
    auto X = input.contiguous().to(torch::kFloat32);

    int64_t M = go.size(0);
    int64_t N = W.size(1);
    int64_t K = W.size(0);

    auto zero_bias_n = torch::zeros({N}, torch::kFloat32);
    auto zero_bias_k = torch::zeros({K}, torch::kFloat32);

    // grad_input = go @ W  (M,N) @ (N,K) = (M,K)
    auto grad_input = std_kernel_dispatch(ocr_k_linear_bwd, go, W, zero_bias_n);

    // grad_weight = go.T @ X  (N,M) @ (M,K) = (N,K)
    auto grad_weight = std_kernel_dispatch(ocr_k_linear_bwd, go.t(), X, zero_bias_k);

    // grad_bias = go.sum(0) (N,)
    auto grad_bias = go.sum(0);

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

torch::Tensor relu_backward(torch::Tensor grad_output, torch::Tensor input) {
    ensure_init();
    auto go = grad_output.contiguous().to(torch::kFloat32);
    auto X = input.contiguous().to(torch::kFloat32);
    int64_t n = X.numel();
    if (!ocr_initialized || !ocr_k_relu_bwd || !ocr_queue) {
        auto mask = (X > 0).to(torch::kFloat32);
        return go * mask;
    }
    auto out = torch::empty_like(go);
    const float* gp = go.data_ptr<float>();
    const float* xp = X.data_ptr<float>();
    float* op = out.data_ptr<float>();
    cl_mem dG = alloc_buffer((size_t)n * sizeof(float));
    cl_mem dX = alloc_buffer((size_t)n * sizeof(float));
    cl_mem dO = alloc_buffer((size_t)n * sizeof(float));
    clEnqueueWriteBuffer(ocr_queue, dG, CL_TRUE, 0, (size_t)n * sizeof(float), gp, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(ocr_queue, dX, CL_TRUE, 0, (size_t)n * sizeof(float), xp, 0, nullptr, nullptr);
    run_binary_unary_kernel(ocr_k_relu_bwd, dG, dX, dO, (int)n);
    clFinish(ocr_queue);
    clEnqueueReadBuffer(ocr_queue, dO, CL_TRUE, 0, (size_t)n * sizeof(float), op, 0, nullptr, nullptr);
    clReleaseMemObject(dG);
    clReleaseMemObject(dX);
    clReleaseMemObject(dO);
    return out;
}

torch::Tensor tanh_backward(torch::Tensor grad_output, torch::Tensor output) {
    ensure_init();
    auto go = grad_output.contiguous().to(torch::kFloat32);
    auto Y = output.contiguous().to(torch::kFloat32);
    int64_t n = Y.numel();
    if (!ocr_initialized || !ocr_k_tanh_bwd || !ocr_queue) {
        return go * (1.0f - Y * Y);
    }
    auto out = torch::empty_like(go);
    const float* gp = go.data_ptr<float>();
    const float* yp = Y.data_ptr<float>();
    float* op = out.data_ptr<float>();
    cl_mem dG = alloc_buffer((size_t)n * sizeof(float));
    cl_mem dY = alloc_buffer((size_t)n * sizeof(float));
    cl_mem dO = alloc_buffer((size_t)n * sizeof(float));
    clEnqueueWriteBuffer(ocr_queue, dG, CL_TRUE, 0, (size_t)n * sizeof(float), gp, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(ocr_queue, dY, CL_TRUE, 0, (size_t)n * sizeof(float), yp, 0, nullptr, nullptr);
    run_binary_unary_kernel(ocr_k_tanh_bwd, dG, dY, dO, (int)n);
    clFinish(ocr_queue);
    clEnqueueReadBuffer(ocr_queue, dO, CL_TRUE, 0, (size_t)n * sizeof(float), op, 0, nullptr, nullptr);
    clReleaseMemObject(dG);
    clReleaseMemObject(dY);
    clReleaseMemObject(dO);
    return out;
}

PYBIND11_MODULE(opencl_ocl, m) {
    m.doc() = "PyTorch OpenCL custom operator extension (torch::Tensor API, tiled GEMM + fused activations + backward)";

    m.def("linear_forward", &linear_forward, "Linear forward (OpenCL, Y = X@W^T + B)");
    m.def("linear_relu_forward", &linear_relu_forward, "Fused Linear + ReLU (OpenCL)");
    m.def("linear_tanh_forward", &linear_tanh_forward, "Fused Linear + Tanh (OpenCL)");
    m.def("relu_forward", &relu_forward, "ReLU forward (OpenCL)");
    m.def("tanh_forward", &tanh_forward, "Tanh forward (OpenCL)");
    m.def("linear_backward", &linear_backward, "Linear backward (OpenCL)");
    m.def("relu_backward", &relu_backward, "ReLU backward (OpenCL)");
    m.def("tanh_backward", &tanh_backward, "Tanh backward (OpenCL)");
    m.def("cleanup", &cleanup_ocl, "Cleanup OpenCL resources");
    m.def("is_available", []() { ensure_init(); return ocr_initialized; }, "Check if OpenCL is available");

    m.attr("__library_version__") = "opencl_ocl-2.3.0";
}
