cmake_minimum_required(VERSION 3.6.2)
project(fastDAS)


message(STATUS "Targetting compiler: ${CMAKE_C_COMPILER}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -flto")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
include_directories(${CMAKE_SOURCE_DIR}/../include)
add_library(das SHARED das.cpp)

