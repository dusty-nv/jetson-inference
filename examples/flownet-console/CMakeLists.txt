
file(GLOB flownetConsoleSources *.cpp)
file(GLOB flownetConsoleIncludes *.h )

cuda_add_executable(flownet-console ${flownetConsoleSources})

target_link_libraries(flownet-console jetson-inference)

install(TARGETS flownet-console DESTINATION bin)
