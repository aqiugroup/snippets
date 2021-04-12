aux_source_directory(. EXECUTABLE)
set(CMAKE_BUILD_TYPE Debug)


aux_source_directory(src DIR_SRCS)
add_executable(
    cpp
    ${DIR_SRCS}
    )

#打印调试信息
MESSAGE(STATUS "_____________________________")
MESSAGE(STATUS "DIR_SRCS: ${DIR_SRCS}")
MESSAGE(STATUS "_____________________________")

target_link_libraries(
    cpp
    common
    )
