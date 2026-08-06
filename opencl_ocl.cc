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

// The linear GEMM is a simple, verified 2D-tiled SGEMM (TILE=16, local-memory
// tiling) with bias and a fused activation. The activation is hard-coded into
// three separate kernel sources (instead of -D macros) because some OpenCL
// drivers cache programs by source content and ignore build options.
// (Tiling/activation design inspired by dlprimitives, MIT, Artyom Beilis.)

namespace {

std::mutex ocl_init_mutex;
cl_context ocr_ctx = nullptr;
cl_command_queue ocr_queue = nullptr;
cl_device_id ocr_device = nullptr;
cl_kernel ocr_k_linear = nullptr;     // Y = X@W^T + B
cl_kernel ocr_k_linear_relu = nullptr;
cl_kernel ocr_k_linear_tanh = nullptr;
cl_kernel ocr_k_relu = nullptr;       // unary elementwise
cl_kernel ocr_k_tanh = nullptr;
bool ocr_initialized = false;

// Cache of parameter buffers (weights / bias). Keyed by host data pointer so the
// same nn.Parameter keeps its cl_mem across forward passes. We always re-upload
// the contents because the optimizer mutates the storage in place during
// training; this avoids re-allocating device memory every step.
std::unordered_map<void*, std::pair<cl_mem, size_t>> g_param_cache;

// Kernel source split into head (computes Y = sum + bias) and tail (closing
// brace) so we can assemble three activation variants.
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

// No activation, relu, tanh variants (the activation line post-processes Y).
const std::string SRC_LINEAR      = sgemm_src("// no activation");
const std::string SRC_LINEAR_RELU = sgemm_src("if (row < M && col < N) Y[row*N + col] = (Y[row*N + col] > 0.0f) ? Y[row*N + col] : 0.0f;");
const std::string SRC_LINEAR_TANH = sgemm_src("if (row < M && col < N) Y[row*N + col] = tanh(Y[row*N + col]);");

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
    // sgemm(W, X, B, Y, M, K, N)
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

torch::Tensor linear_kernel_dispatch(cl_kernel k, torch::Tensor weight, torch::Tensor bias, torch::Tensor input) {
    ensure_init();
    auto W = weight.contiguous().to(torch::kFloat32);
    auto B = bias.contiguous().to(torch::kFloat32);
    auto X = input.contiguous().to(torch::kFloat32);

    int64_t M = X.size(0);
    int64_t K = W.size(1);
    int64_t N = W.size(0);

    if (!ocr_initialized || !k || !ocr_queue) {
        // CPU fallback (ATen)
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
    if (ocr_queue) clReleaseCommandQueue(ocr_queue);
    if (ocr_ctx) clReleaseContext(ocr_ctx);
    ocr_k_linear = nullptr;
    ocr_k_linear_relu = nullptr;
    ocr_k_linear_tanh = nullptr;
    ocr_k_relu = nullptr;
    ocr_k_tanh = nullptr;
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

PYBIND11_MODULE(opencl_ocl, m) {
    m.doc() = "PyTorch OpenCL custom operator extension (torch::Tensor API, tiled GEMM + fused activations)";

    m.def("linear_forward", &linear_forward, "Linear forward (OpenCL, Y = X@W^T + B)");
    m.def("linear_relu_forward", &linear_relu_forward, "Fused Linear + ReLU (OpenCL)");
    m.def("linear_tanh_forward", &linear_tanh_forward, "Fused Linear + Tanh (OpenCL)");
    m.def("relu_forward", &relu_forward, "ReLU forward (OpenCL)");
    m.def("tanh_forward", &tanh_forward, "Tanh forward (OpenCL)");
    m.def("cleanup", &cleanup_ocl, "Cleanup OpenCL resources");
    m.def("is_available", []() { ensure_init(); return ocr_initialized; }, "Check if OpenCL is available");

    m.attr("__library_version__") = "opencl_ocl-2.2.0";
}
