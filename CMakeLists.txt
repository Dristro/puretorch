cmake_minimum_required(VERSION 4.0.0)
project(PureTorch VERSION 1.2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include all the src dirs (so headers work)
include_directories(${CMAKE_SOURCE_DIR}/lib)  # wil remove this later, using custom lib
include_directories(${CMAKE_SOURCE_DIR}/autograd/cpp)

# Collect all .cpp files under dirs
file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/autograd/cpp/*.cpp")

add_executable(puretorch ${SOURCES})
