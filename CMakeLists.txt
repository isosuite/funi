cmake_minimum_required(VERSION 3.18)
project(
  funi
  VERSION 0.1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimized by default (-O3 -DNDEBUG / /O2 /DNDEBUG via CMake's Release flags)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(funi NB_STATIC NOMINSIZE src/module.cpp)
target_include_directories(funi PRIVATE src)

install(TARGETS funi LIBRARY DESTINATION .)
