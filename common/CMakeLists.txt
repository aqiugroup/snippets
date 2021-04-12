aux_source_directory(. COMMON)
set(CMAKE_BUILD_TYPE Debug)
add_library(common ${COMMON})

message(STATUS "common library: " ${COMMON})