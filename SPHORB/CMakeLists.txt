cmake_minimum_required(VERSION 2.8.3)
project(SPHORB)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-result")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-misleading-indentation -Wno-reorder -Wno-unused-function -Wno-sign-compare -Wno-class-memaccess")

set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV 4 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
    roslib  
)

set(SPHORB_SOURCES
    pfm.cpp 
    utility.cpp 
    detector.cpp 
    nonmax.cpp 
    SPHORB.cpp
)

add_library(sphorb_lib STATIC ${SPHORB_SOURCES})

target_include_directories(sphorb_lib PUBLIC 
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(sphorb_lib PUBLIC ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})