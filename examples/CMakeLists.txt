
add_subdirectory(imagenet-console)
add_subdirectory(imagenet-camera)

add_subdirectory(detectnet-console)
add_subdirectory(detectnet-camera)

add_subdirectory(segnet-console)
add_subdirectory(segnet-camera)

add_subdirectory(superres-console)

if(HAS_OPENCV)
	add_subdirectory(homography-console)
	add_subdirectory(homography-camera)
endif()
