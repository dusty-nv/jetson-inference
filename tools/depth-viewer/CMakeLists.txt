
if(BUILD_EXPERIMENTAL)
	file(GLOB depthViewerSources *.cpp)
	file(GLOB depthViewerIncludes *.h )

	cuda_add_executable(depth-viewer ${depthViewerSources})

	target_link_libraries(depth-viewer jetson-inference)

	install(TARGETS depth-viewer DESTINATION bin)
endif()
