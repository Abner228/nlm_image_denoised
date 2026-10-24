cmake_minimum_required(VERSION 3.10)

project(OpenCVExample LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found, version: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "CUDA found, version: ${CUDA_VERSION}")
    message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
else()
    message(FATAL_ERROR "CUDA not found")
endif()

add_executable(opencv_example src/main.cpp src/nlm.cu)

target_include_directories(opencv_example PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    include
)

target_link_libraries(opencv_example PRIVATE
    ${OpenCV_LIBS} 
    ${CUDA_LIBRARIES}
)