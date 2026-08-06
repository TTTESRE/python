cmake_minimum_required(VERSION 3.18)
project(opencl_ocl)

find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenCL REQUIRED)

# The pybind11 type caster for torch::Tensor lives in torch_python, which is not
# pulled in by the `torch` target. Locate and link it explicitly.
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
    OUTPUT_VARIABLE TORCH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
find_library(TORCH_PYTHON_LIBRARY NAMES torch_python PATHS ${TORCH_LIB_DIR} NO_DEFAULT_PATH)

pybind11_add_module(opencl_ocl opencl_ocl.cc)
target_link_libraries(opencl_ocl PRIVATE torch ${TORCH_PYTHON_LIBRARY} OpenCL::OpenCL)
target_compile_options(opencl_ocl PRIVATE -O3 -std=c++17)
