
add_subdirectory(imagenet-console)
add_subdirectory(imagenet-camera)

add_subdirectory(detectnet-console)
add_subdirectory(detectnet-camera)

add_subdirectory(depthnet-console)
add_subdirectory(depthnet-camera)

add_subdirectory(flownet-console)

add_subdirectory(segnet-console)
add_subdirectory(segnet-camera)

if(CUDA_VERSION_MAJOR GREATER 9)
	add_subdirectory(homography-console)
	add_subdirectory(homography-camera)

	add_subdirectory(superres-console)
endif()
