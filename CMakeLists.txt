cmake_minimum_required(VERSION 3.15)
project(achmm_trellis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(achmm_trellis
    src/trellis.cpp
    src/bindings.cpp
)

target_include_directories(achmm_trellis PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(MSVC)
    target_compile_options(achmm_trellis PRIVATE /O2 /W4)
else()
    target_compile_options(achmm_trellis PRIVATE -O3 -Wall -Wextra)
endif()
