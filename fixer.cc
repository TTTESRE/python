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
#include <sstream>
#include <thread>
#include <chrono>

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
    // Trim whitespace
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r' || result.back() == ' '))
        result.pop_back();
    return result;
}

bool gdb_force_render(pid_t pid) {
    std::string gdb_cmd = "gdb -p " + std::to_string(pid) +
        " -batch -ex \"call (int)PyRun_SimpleString(\\\"import pygame; pygame.display.flip(); pygame.event.pump()\\\")\" 2>/dev/null";
    int ret = system(gdb_cmd.c_str());
    return ret == 0;
}

bool gdb_force_render_all(pid_t pid) {
    std::string script = R"(
import pygame
pygame.display.flip()
pygame.event.pump()
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
    std::cerr << "  Modes: fixrun, fixtrain, fixruntrain" << std::endl;
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

    std::cout << "Fixer attached to PID " << target_pid << " in " << mode << " mode" << std::endl;
    std::cout << "Forcing pygame render every 33ms (30 FPS)..." << std::endl;

    while (true) {
        // Check if process is still alive
        if (kill(target_pid, 0) != 0) {
            std::cerr << "Target process " << target_pid << " is no longer running." << std::endl;
            break;
        }

        if (!gdb_force_render_all(target_pid)) {
            // GDB call failed, try alternative: send SIGUSR1 to trigger render
            kill(target_pid, SIGUSR1);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }

    return 0;
}