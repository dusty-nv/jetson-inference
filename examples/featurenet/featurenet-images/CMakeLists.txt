
file(GLOB featurenetImagesSources *.cpp)
file(GLOB featurenetImagesIncludes *.h )

cuda_add_executable(featurenet-images ${featurenetImagesSources})
target_link_libraries(featurenet-images jetson-inference)
install(TARGETS featurenet-images DESTINATION bin)
