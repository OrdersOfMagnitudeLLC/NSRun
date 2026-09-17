cmake_minimum_required(VERSION 3.20)
project(nsquant CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(nsquant
    nsquant.cpp
    gguf_parser.cpp
    tokenizer.cpp
    nsm_loader.cpp
)
target_include_directories(nsquant PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
find_library(Z_LIB NAMES z libz zlib)
if(Z_LIB)
    target_link_libraries(nsquant PRIVATE ${Z_LIB} Threads::Threads OpenMP::OpenMP_CXX)
else()
    target_link_libraries(nsquant PRIVATE Threads::Threads OpenMP::OpenMP_CXX)
endif()
target_compile_options(nsquant PRIVATE -O3 -march=native -fopenmp)
