
cmake_minimum_required(VERSION 2.8)
project(jetson-inference)

# submodule warning
message(" ")
message("Note:  this project uses git submodules in the source tree.")
message("       if you haven't already, run the following command from")
message("       the project's root directory:")
message(" ")
message("           git submodule update --init") 
message("\n")

if( NOT EXISTS "${PROJECT_SOURCE_DIR}/utils/.git" )
	message("Note:  required git submodules have not been detected.")
	message("       first, please run the following command from the")
	message("       the project's root directory to clone them:")
	message(" ")
	message("          git submodule update --init")
	message(" ")
	message(FATAL_ERROR "missing required git submodules, see instructions above")
endif()


# setup tensorRT flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	# -std=gnu++11
set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")


# if this is the first time running cmake, perform pre-build dependency install script (or if the user manually triggers re-building the dependencies)
if( ${BUILD_DEPS} )
	message("-- Launching pre-build dependency installer script...")

	execute_process(COMMAND sh ../CMakePreBuild.sh
				WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
				RESULT_VARIABLE PREBUILD_SCRIPT_RESULT)

	set(BUILD_DEPS "NO" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed." FORCE)
	message("-- Finished installing dependencies")
endif()


# setup CUDA
find_package(CUDA)
message("-- CUDA version: ${CUDA_VERSION}")

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS}; 
    -O3 
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_62,code=sm_62
)

if(CUDA_VERSION_MAJOR GREATER 9)
	message("-- CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_72")

	set(
		CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS}; 
		-gencode arch=compute_72,code=sm_72
	)

	# OpenCV used for findHomography() and decomposeHomography()
	# OpenCV version >= 3.0.0 required for decomposeHomography()
	find_package(OpenCV 3.0.0 COMPONENTS core calib3d REQUIRED)
endif()

# setup project output paths
set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_PROCESSOR})
set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)

file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)

message("-- system arch:  ${CMAKE_SYSTEM_PROCESSOR}")
message("-- output path:  ${PROJECT_OUTPUT_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)


# build C/C++ interface
include_directories(${PROJECT_INCLUDE_DIR} ${GIE_PATH}/include)
include_directories(/usr/include/gstreamer-1.0 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include /usr/include/glib-2.0 /usr/include/libxml2 /usr/lib/aarch64-linux-gnu/glib-2.0/include/)

file(GLOB inferenceSources *.cpp *.cu calibration/*.cpp plugins/*.cpp)
file(GLOB inferenceIncludes *.h calibration/*.h)

cuda_add_library(jetson-inference SHARED ${inferenceSources})
#target_link_libraries(jetson-inference nvcaffe_parser nvinfer)		# gstreamer-0.10 gstbase-0.10 gstapp-0.10 


# transfer all headers to the include directory
foreach(include ${inferenceIncludes})
	message("-- Copying ${include}")
	configure_file(${include} ${PROJECT_INCLUDE_DIR} COPYONLY)
endforeach()


# create symbolic link for network data
execute_process( COMMAND "${CMAKE_COMMAND}" "-E" "create_symlink" "${PROJECT_SOURCE_DIR}/data/networks" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/networks" )
  
  
# copy image data
file(GLOB imageData ${PROJECT_SOURCE_DIR}/data/images/*)

foreach(image ${imageData})
	message("-- Copying ${image}")
	file(COPY ${image} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
	#configure_file(${include} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COPYONLY)
endforeach()


# build subdirectories
add_subdirectory(docs)
add_subdirectory(examples)
add_subdirectory(tools)
add_subdirectory(utils)
add_subdirectory(python)


# set linker options
target_link_libraries(jetson-inference jetson-utils nvinfer nvcaffe_parser)

if(CUDA_VERSION_MAJOR GREATER 9)
	target_link_libraries(jetson-inference nvonnxparser opencv_core opencv_calib3d)
endif()


# install includes
foreach(include ${inferenceIncludes})
    install(FILES "${include}" DESTINATION include/jetson-inference)
endforeach()

# install symlink to networks
install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/data/networks ${CMAKE_INSTALL_PREFIX}/bin/networks )" )

# install the shared library
install(TARGETS jetson-inference DESTINATION lib EXPORT jetson-inferenceConfig)

# install the cmake project, for importing
install(EXPORT jetson-inferenceConfig DESTINATION share/jetson-inference/cmake)

# run ldconfig after installing
install(CODE "execute_process( COMMAND ldconfig )")

