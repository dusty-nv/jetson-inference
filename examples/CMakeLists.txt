
add_subdirectory(imagenet)
add_subdirectory(depthnet)
add_subdirectory(detectnet)
add_subdirectory(posenet)
add_subdirectory(segnet)
add_subdirectory(actionnet)
add_subdirectory(backgroundnet)

# experimental examples
if(BUILD_EXPERIMENTAL)
	add_subdirectory(experimental/featurenet)
		
	add_subdirectory(experimental/flownet-console)
	add_subdirectory(experimental/flownet-camera)

	add_subdirectory(experimental/odometry-console)
	add_subdirectory(experimental/odometry-plot)

	add_subdirectory(experimental/stereonet-console)
	add_subdirectory(experimental/stereonet-camera)

	add_subdirectory(experimental/superres-console)

	if(HAS_OPENCV)
		add_subdirectory(experimental/homography-console)
		add_subdirectory(experimental/homography-camera)
	endif()
endif()
