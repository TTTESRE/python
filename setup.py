import os
import sys
import glob
import shutil
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False
except Exception:
    bdist_wheel = None

class CMakeBuildExt(build_ext):
    def run(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.join(script_dir, "build")
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        os.makedirs(build_dir, exist_ok=True)

        # Try to get pybind11 cmake dir; fall back to system path
        try:
            pybind11_cmake = subprocess.run(
                [sys.executable, "-m", "pybind11", "--cmakedir"],
                capture_output=True, text=True
            ).stdout.strip()
        except Exception:
            pybind11_cmake = "/usr/lib/cmake/pybind11"

        # Try to get torch cmake path; fall back to /home/user/libtorch
        try:
            torch_cmake = subprocess.run(
                [sys.executable, "-c", "import torch; print(torch.utils.cmake_prefix_path)"],
                capture_output=True, text=True
            ).stdout.strip()
        except Exception:
            torch_cmake = "/home/user/libtorch"

        env = os.environ.copy()
        env["LIBTORCH"] = "/home/user/libtorch"
        env["CMAKE_PREFIX_PATH"] = torch_cmake + ";" + pybind11_cmake

        cmake_cmd = [
            "cmake",
            "-DCMAKE_PREFIX_PATH=" + torch_cmake + ";" + pybind11_cmake,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            script_dir,
        ]
        print(f"Running cmake in {build_dir} with CMAKE_PREFIX_PATH={env.get('CMAKE_PREFIX_PATH')}")
        result = subprocess.run(cmake_cmd, cwd=build_dir, env=env, capture_output=True, text=True)
        print("CMAKE returncode:", result.returncode)
        print("CMAKE STDOUT:", result.stdout)
        print("CMAKE STDERR:", result.stderr)
        if result.returncode != 0:
            raise RuntimeError("CMake configuration failed")

        build_cmd = ["cmake", "--build", build_dir, "--config", "Release", "-j", str(os.cpu_count())]
        result = subprocess.run(build_cmd, cwd=build_dir, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise RuntimeError("CMake build failed")

        pkg_dir = os.path.join(script_dir, "opencl_ocl")
        os.makedirs(pkg_dir, exist_ok=True)
        so_pattern = os.path.join(build_dir, "opencl_ocl*.so")
        so_files = glob.glob(so_pattern)
        if not so_files:
            raise RuntimeError(f"Built extension not found: {so_pattern}")
        for so_src in so_files:
            so_dst = os.path.join(pkg_dir, os.path.basename(so_src))
            shutil.copy2(so_src, so_dst)
            print(f"Copied {so_src} -> {so_dst}")

setup(
    name="opencl-ocl",
    version="0.1.1",
    description="OpenCL acceleration for PyTorch neural network operations via libtorch tensors",
    long_description=open("README").read() if os.path.exists("README") else "",
    long_description_content_type="text/markdown",
    author="tttesre",
    author_email="ttes2@proton.me",
    packages=find_packages(include=["opencl_ocl"]),
    package_data={"opencl_ocl": ["*.so"]},
    cmdclass={"build_ext": CMakeBuildExt, **({"bdist_wheel": bdist_wheel} if bdist_wheel else {})},
    python_requires=">=3.10",
    install_requires=["torch>=2.0"],
)
