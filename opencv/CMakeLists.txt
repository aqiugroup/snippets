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

include_directories(${OpenCv_INCLUDE_DIRS})

# set(
#     OpenCV_LIBS
#     -lopencv_contrib
#     -lopencv_core
#     -lopencv_highgui
#     -lopencv_imgproc
#     -lopencv_legacy
#     )

# aux_source_directory(common DIR_COMMON)
aux_source_directory(src DIR_SRCS)
# target_include_directories(
#     hello-opencv PUBLIC
#     src
#     )
# 以上两种是类似的，第一种更加的通用

add_executable(
    hello-opencv
    ${DIR_SRCS}
    ${DIR_COMMON}
    )
# add_executable(
#     hello-opencv
#     src/main.cpp
#     src/img.cpp
#     )
# 以上两种是类似的，第一种更加的通用

target_link_libraries(
    hello-opencv
    ${OpenCV_LIBS}
    common
    )


# foreach(T_FILE_NAME ${EXECUTABLE})
#     get_filename_component(EXECUTABLE_NAME ${T_FILE_NAME} NAME_WE)
#     message(STATUS "test：" ${EXECUTABLE_NAME} " T_FILE : "  ${T_FILE_NAME} )
#     add_executable(${EXECUTABLE_NAME} ${T_FILE_NAME})
#     target_link_libraries(
#         ${EXECUTABLE_NAME}
#         PRIVATE
#         common
#         ${CMAKE_THREAD_LIBS_INIT}
#         # GTest::GTest
#         # GTest::Main
#     )
#     # gtest_discover_tests(${EXECUTABLE_NAME})
# endforeach()
