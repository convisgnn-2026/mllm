cmake_minimum_required(VERSION 3.12)
project(cpp_optimizer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Use local pybind11 from 3rdparty
set(pybind11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/pybind11")
add_subdirectory(${pybind11_DIR} pybind11)

# Add optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
endif()

# Create the pybind11 module
pybind11_add_module(cpp_optimizer pybind_wrapper.cpp)

# Set properties
target_compile_definitions(cpp_optimizer PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_link_libraries(cpp_optimizer PRIVATE pybind11::module)

# Set the output directory
set_target_properties(cpp_optimizer PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
