
file(GLOB depthnetConsoleSources *.cpp)
file(GLOB depthnetConsoleIncludes *.h )

cuda_add_executable(depthnet-console ${depthnetConsoleSources})

target_link_libraries(depthnet-console jetson-inference)

install(TARGETS depthnet-console DESTINATION bin)
