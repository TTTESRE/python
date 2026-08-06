#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <string>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>

// fixer_opencl: watchdog for the OpenCL GEMM kernel.
//
// It attaches GDB to the running text_trainer.py process and, every second,
// runs a Python snippet that validates the OpenCL linear kernel against a CPU
// reference. If the OpenCL result is wrong (driver/device corruption, lost
// context, etc.) it calls opencl_ocl.cleanup() so all future ops transparently
// fall back to the ATen/CPU path. This mirrors fixer.cc (which forces pygame
// display updates) but for the compute backend.

std::string find_python_pid(const std::string& target_arg) {
    std::string cmd = "ps aux | grep '[t]ext_trainer.py' | awk '{print $2}'";
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return "";
    char buf[256];
    std::string result;
    while (fgets(buf, sizeof(buf), fp)) {
        result += buf;
    }
    pclose(fp);
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r' || result.back() == ' '))
        result.pop_back();
    return result;
}

// Validate OpenCL against CPU and disable it on mismatch. Returns true if a fix
// was applied (OpenCL was found broken and disabled).
bool gdb_validate_and_fix(pid_t pid) {
    std::string script = R"(
import torch, opencl_ocl
torch.manual_seed(1234)
bad = False
try:
    M, K, N = 32, 64, 64
    W = torch.randn(N, K); B = torch.randn(N); X = torch.randn(M, K)
    ref = (X @ W.T + B)
    got = opencl_ocl.linear_forward(W, B, X)
    if got.shape != ref.shape:
        bad = True
    else:
        err = (ref.float() - got.float()).abs().max().item()
        if err > 1e-2:
            bad = True
        # also exercise fused kernels
        rref = torch.relu(X @ W.T + B)
        rgot = opencl_ocl.linear_relu_forward(W, B, X)
        if (rref.float() - rgot.float()).abs().max().item() > 1e-2:
            bad = True
        tref = torch.tanh(X @ W.T + B)
        tgot = opencl_ocl.linear_tanh_forward(W, B, X)
        if (tref.float() - tgot.float()).abs().max().item() > 1e-2:
            bad = True
except Exception as e:
    bad = True
    print('[fixer-opencl] validation exception:', e)
if bad:
    opencl_ocl.cleanup()
    print('[fixer-opencl] OpenCL output mismatch -> disabled (CPU fallback active)')
else:
    print('[fixer-opencl] OpenCL OK')
)";
    std::string escaped;
    for (char c : script) {
        if (c == '"' || c == '\\') escaped += '\\';
        escaped += c;
    }
    std::string gdb_cmd = "gdb -p " + std::to_string(pid) +
        " -batch -ex \"call (int)PyRun_SimpleString(\\\"" + escaped + "\\\")\" 2>/dev/null";
    int ret = system(gdb_cmd.c_str());
    return ret == 0;
}

void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <mode> [pid]" << std::endl;
    std::cerr << "  Modes: fixrun, fixtrain, fixruntrain, fixopencl, opencl" << std::endl;
    std::cerr << "  If pid is not given, finds text_trainer.py process automatically" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];
    pid_t target_pid = 0;
    if (argc >= 3) {
        target_pid = (pid_t)std::atoi(argv[2]);
    }
    if (target_pid == 0) {
        std::string pid_str = find_python_pid(mode);
        if (pid_str.empty()) {
            std::cerr << "No text_trainer.py process found. Start it first." << std::endl;
            return 1;
        }
        target_pid = (pid_t)std::atoi(pid_str.c_str());
    }

    std::cout << "OpenCL fixer attached to PID " << target_pid << " in " << mode << " mode" << std::endl;
    std::cout << "Validating OpenCL GEMM every 1s (disables on mismatch)..." << std::endl;

    while (true) {
        if (kill(target_pid, 0) != 0) {
            std::cerr << "Target process " << target_pid << " is no longer running." << std::endl;
            break;
        }
        gdb_validate_and_fix(target_pid);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}
