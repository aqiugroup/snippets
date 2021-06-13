aux_source_directory(. EXECUTABLE)
set(CMAKE_BUILD_TYPE Debug)

# 添加Eigen头文件
# ubuntu
# include_directories( "/usr/include/eigen3" )
# macos
include_directories( "/usr/local/include/eigen3" )

# aux_source_directory(ch1 DIR_SRCS)
# aux_source_directory(ch2 DIR_SRCS)

# aux_source_directory(ch3/useEigen DIR_SRCS)
# add_executable(
#     # ch1
#     # ch2
#     ch3
#     ${DIR_SRCS}
#     )

# #打印调试信息
# MESSAGE(STATUS "_____________________________")
# MESSAGE(STATUS "DIR_SRCS: ${DIR_SRCS}")
# MESSAGE(STATUS "_____________________________")

# target_link_libraries(
#     # ch1
#     # ch2
#     ch3
#     common
#     )

add_subdirectory(ch1)
add_subdirectory(ch2)

add_subdirectory(ch3/useEigen)
add_subdirectory(ch3/useGeometry)
add_subdirectory(ch3/visualizeGeometry)
add_subdirectory(ch4/useSophus)
add_subdirectory(ch5/imageBasics)
# add_subdirectory(ch5/joinMap)
add_subdirectory(ch6/ceres_curve_fitting)
add_subdirectory(ch6/g2o_curve_fitting)
add_subdirectory(ch7)
add_subdirectory(ch8/directMethod)
add_subdirectory(ch8/LKFlow)



