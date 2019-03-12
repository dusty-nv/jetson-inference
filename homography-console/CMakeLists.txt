
file(GLOB homographyConsoleSources *.cpp)
file(GLOB homographyConsoleIncludes *.h )

cuda_add_executable(homography-console ${homographyConsoleSources})

target_link_libraries(homography-console jetson-inference)

install(TARGETS homography-console DESTINATION bin)
