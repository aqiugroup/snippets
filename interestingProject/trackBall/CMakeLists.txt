aux_source_directory(. EXECUTABLE)
set(CMAKE_BUILD_TYPE Debug)

# cmake_minimum_required(VERSION 2.8)
find_package(OpenCV REQUIRED) # OpenCV

#打印调试信息
MESSAGE(STATUS "_____________________________")
MESSAGE(STATUS "Project: ${PROJECT_NAME}")
MESSAGE(STATUS "OpenCV library status:")
MESSAGE(STATUS "    version: ${OpenCV_VERSION}")
MESSAGE(STATUS "    libraries: ${OpenCV_LIBS}")
MESSAGE(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "_____________________________")

include_directories(${OpenCV_INCLUDE_DIRS})
aux_source_directory(src DIR_SRCS)
# target_include_directories(
#     hello-opencv PUBLIC
#     src
#     )
# 以上两种是类似的，第一种更加的通用

add_executable(
    trackBall
    ${DIR_SRCS}
    )
# add_executable(
#     hello-opencv
#     src/main.cpp
#     src/img.cpp
#     )
# 以上两种是类似的，第一种更加的通用

target_link_libraries(
    trackBall
    ${OpenCV_LIBS}
    common
    )