
file(GLOB odometryPlotSources *.cpp)
file(GLOB odometryPlotIncludes *.h )

cuda_add_executable(odometry-plot ${odometryPlotSources})

target_link_libraries(odometry-plot jetson-inference)

install(TARGETS odometry-plot DESTINATION bin)
