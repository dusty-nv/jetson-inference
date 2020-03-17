
add_subdirectory(imagenet-console)
add_subdirectory(imagenet-camera)

add_subdirectory(detectnet-console)
add_subdirectory(detectnet-camera)

add_subdirectory(segnet-console)
add_subdirectory(segnet-camera)

# experimental examples
if(BUILD_EXPERIMENTAL)
	add_subdirectory(experimental/depthnet-console)
	add_subdirectory(experimental/depthnet-camera)

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
