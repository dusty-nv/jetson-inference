
file(GLOB odometryConsoleSources *.cpp)
file(GLOB odometryConsoleIncludes *.h )

cuda_add_executable(odometry-console ${odometryConsoleSources})

target_link_libraries(odometry-console jetson-inference)

install(TARGETS odometry-console DESTINATION bin)
