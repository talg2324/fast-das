message(STATUS "Targetting compiler: ${CMAKE_CXX_COMPILER}")
set(CMAKE_CXX_STANDARD 17)
add_library(das SHARED das.cpp)
target_compile_options(das PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
                       -O3
                       -fPIC
                       >)