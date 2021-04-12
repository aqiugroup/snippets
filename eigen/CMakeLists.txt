aux_source_directory(. EXECUTABLE)
set(CMAKE_BUILD_TYPE Debug)

# 添加Eigen头文件
# ubuntu
# include_directories( "/usr/include/eigen3" )
# macos
include_directories( "/usr/local/include/eigen3" )

aux_source_directory(src DIR_SRCS)
add_executable(
    eigen-test
    ${DIR_SRCS}
    )

#打印调试信息
MESSAGE(STATUS "_____________________________")
MESSAGE(STATUS "DIR_SRCS: ${DIR_SRCS}")
MESSAGE(STATUS "_____________________________")

target_link_libraries(
    eigen-test
    common
    )
