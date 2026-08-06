#!/usr/bin/env python3
"""
Setup script for compiling the OpenCL extension using cmake + libtorch
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def find_libtorch():
    common_paths = [
        "/usr/local/libtorch",
        "/usr/lib/libtorch",
        os.path.expanduser("~/libtorch"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "libtorch"),
        "/home/user/libtorch",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    return None

def main():
    libtorch_path = find_libtorch()
    if not libtorch_path:
        print("Error: Could not find libtorch installation")
        print("Set LIBTORCH environment variable to its path")
        sys.exit(1)

    print(f"Found libtorch at: {libtorch_path}")

    script_dir = Path(__file__).parent.absolute()
    build_dir = script_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    env = os.environ.copy()
    env["LIBTORCH"] = libtorch_path
    env["CMAKE_PREFIX_PATH"] = libtorch_path

    pybind11_dir = subprocess.run(
        [sys.executable, "-m", "pybind11", "--cmakedir"],
        capture_output=True, text=True
    ).stdout.strip()

    torch_cmake_dir = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.utils.cmake_prefix_path)"],
        capture_output=True, text=True
    ).stdout.strip()

    cmake_cmd = [
        "cmake",
        "-DCMAKE_PREFIX_PATH=" + torch_cmake_dir + ";" + pybind11_dir,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DPYTHON_EXECUTABLE=" + sys.executable,
        str(script_dir),
    ]

    print("Running CMake...")
    result = subprocess.run(cmake_cmd, cwd=build_dir, env=env, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("CMake stderr:", result.stderr)
        sys.exit(1)

    print("Running build...")
    result = subprocess.run(
        ["cmake", "--build", ".", "--config", "Release", "-j", str(os.cpu_count())],
        cwd=build_dir, env=env, capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("Build stderr:", result.stderr)
        sys.exit(1)

    so_name = "opencl_ocl.cpython-312-x86_64-linux-gnu.so"
    src = build_dir / so_name
    dst = script_dir / so_name

    if src.exists():
        shutil.copy2(src, dst)
        print(f"Successfully built extension to {dst}")
    else:
        for f in build_dir.rglob("*.so"):
            shutil.copy2(f, dst)
            print(f"Copied {f} to {dst}")
            return
        print("Error: Compiled extension not found at", src)
        sys.exit(1)

if __name__ == "__main__":
    main()