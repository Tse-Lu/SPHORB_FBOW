cmake_minimum_required(VERSION 3.0)
project(sporb_fbow_train LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Suppress warnings from third-party code
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-all -Wno-extra")

# -------------------------------------------------------------------
# OpenCV 4
# -------------------------------------------------------------------
find_package(OpenCV 4 REQUIRED)

# -------------------------------------------------------------------
# SPHORB static library (built without catkin/ROS)
# -------------------------------------------------------------------
# A stub ros/package.h is provided under stubs/ to satisfy the unused
# #include <ros/package.h> in SPHORB.cpp.

set(SPHORB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SPHORB)

add_library(sphorb STATIC
    ${SPHORB_DIR}/pfm.cpp
    ${SPHORB_DIR}/utility.cpp
    ${SPHORB_DIR}/detector.cpp
    ${SPHORB_DIR}/nonmax.cpp
    ${SPHORB_DIR}/SPHORB.cpp
)

target_include_directories(sphorb PUBLIC
    ${SPHORB_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(sphorb PUBLIC ${OpenCV_LIBRARIES})

# -------------------------------------------------------------------
# FBOW static library (core only, no utils/tests)
# -------------------------------------------------------------------
set(FBOW_DIR ${CMAKE_CURRENT_SOURCE_DIR}/FBOW)

add_library(fbow STATIC
    ${FBOW_DIR}/src/fbow.cpp
    ${FBOW_DIR}/src/vocabulary_creator.cpp
)

target_include_directories(fbow PUBLIC
    ${FBOW_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(fbow PUBLIC ${OpenCV_LIBRARIES})

# FBOW uses AVX/SSE3 SIMD intrinsics for distance computation
target_compile_options(fbow PRIVATE -mavx -msse3)
target_compile_definitions(fbow PRIVATE USE_AVX USE_SSE3)

# FBOW needs pthreads for std::thread
find_package(Threads REQUIRED)
target_link_libraries(fbow PUBLIC Threads::Threads)

# Optional OpenMP support (FBOW detects this via compile definitions)
find_package(OpenMP)
if(OpenMP_FOUND OR OpenMP_CXX_FOUND)
    target_link_libraries(fbow PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(fbow PRIVATE USE_OPENMP)
endif()

# -------------------------------------------------------------------
# Training executable
# -------------------------------------------------------------------
add_executable(train_sporb_fbow train_sporb_fbow.cpp)
target_link_libraries(train_sporb_fbow PRIVATE sphorb fbow ${OpenCV_LIBRARIES})

# -------------------------------------------------------------------
# Print summary
# -------------------------------------------------------------------
message(STATUS "SPHORB source dir: ${SPHORB_DIR}")
message(STATUS "FBOW source dir:   ${FBOW_DIR}")
message(STATUS "OpenCV version:    ${OpenCV_VERSION}")
message(STATUS "OpenMP found:      ${OPENMP_FOUND}")
