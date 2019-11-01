
file(GLOB stereonetConsoleSources *.cpp)
file(GLOB stereonetConsoleIncludes *.h )

cuda_add_executable(stereonet-console ${stereonetConsoleSources})

target_link_libraries(stereonet-console jetson-inference)

install(TARGETS stereonet-console DESTINATION bin)
