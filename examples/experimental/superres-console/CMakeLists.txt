
file(GLOB superresConsoleSources *.cpp)
file(GLOB superresConsoleIncludes *.h )

cuda_add_executable(superres-console ${superresConsoleSources})

target_link_libraries(superres-console jetson-inference)

install(TARGETS superres-console DESTINATION bin)
