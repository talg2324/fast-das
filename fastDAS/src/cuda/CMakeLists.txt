message(STATUS "Targetting compiler: ${CMAKE_CUDA_COMPILER}")
set(CMAKE_CUDA_STANDARD 17)
add_library(das_cu SHARED das.cu)
set_target_properties(das_cu PROPERTIES CUDA_ARCHITECTURES "90")
target_compile_options(das_cu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -O3
                       --use_fast_math
                       --diag-suppress=20039
                       >)