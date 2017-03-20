
cmake_minimum_required(VERSION 2.8)
project(jetson-inference)

# setup tensorRT flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	# -std=gnu++11
set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")


# if this is the first time running cmake, perform pre-build dependency install script (or if the user manually triggers re-building the dependencies)
if( ${BUILD_DEPS} )
	message("Launching pre-build dependency installer script...")

	execute_process(COMMAND sh ../CMakePreBuild.sh
				WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
				RESULT_VARIABLE PREBUILD_SCRIPT_RESULT)

	set(BUILD_DEPS "NO" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed." FORCE)
	message("Finished installing dependencies")
endif()


# Qt is used to load images (installed by ubuntu-desktop)
find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})


# setup CUDA
find_package(CUDA)

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7l")
	message("-- Detected ${CMAKE_SYSTEM_PROCESSOR} assuming TK1")
	set(
		CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS}; 
        	-O3 -gencode arch=compute_32,code=sm_32, -std=c++11 -Xcompiler -D__CORRECT_ISO_CPP11_MATH_H_PROTO 
)
else()
	# TX2 R27.0
	if(CMAKE_SYSTEM STREQUAL "Linux-4.4.15-tegra")
		message("-- Detected ${CMAKE_SYSTEM_PROCESSOR} assuming TX2 (${CMAKE_SYSTEM})")
		set(
			CUDA_NVCC_FLAGS
			${CUDA_NVCC_FLAGS}; 
        		-O3 -gencode arch=compute_62,code=compute_62)
	# TX2 assume strings greater are future releases
	elseif(CMAKE_SYSTEM STRGREATER "Linux-4.4.15-tegra")
		message("-- Detected ${CMAKE_SYSTEM_PROCESSOR} assuming TX2 (${CMAKE_SYSTEM})")
		set(
			CUDA_NVCC_FLAGS
			${CUDA_NVCC_FLAGS}; 
        		-O3 -gencode arch=compute_62,code=compute_62)
	# Older kernel versions assume TX1
	elseif()
		message("-- Detected ${CMAKE_SYSTEM_PROCESSOR} assuming TX1")
		set(
			CUDA_NVCC_FLAGS
			${CUDA_NVCC_FLAGS}; 
        		-O3 -gencode arch=compute_53,code=sm_53 )
	endif()
endif()

# Aravis setup if installed
if(EXISTS "/usr/local/lib/libaravis-0.6.so")
  include_directories(/usr/local/include/aravis-0.6)
  set(ARAVIS_LIB aravis-0.6)
endif()

# setup project output paths
set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_PROCESSOR})
set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)

file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)

message("-- output path:  ${PROJECT_OUTPUT_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)


# build C/C++ interface
include_directories(${PROJECT_INCLUDE_DIR} ${GIE_PATH}/include)
include_directories(/usr/include/gstreamer-1.0 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include /usr/include/glib-2.0 /usr/include/libxml2 /usr/lib/aarch64-linux-gnu/glib-2.0/include/)

file(GLOB inferenceSources *.cpp *.cu camera/*.cpp cuda/*.cu display/*.cpp)
file(GLOB inferenceIncludes *.h camera/*.h cuda/*.h display/*.h)

cuda_add_library(jetson-inference SHARED ${inferenceSources})
target_link_libraries(jetson-inference nvcaffe_parser nvinfer Qt4::QtGui GL GLEW gstreamer-1.0 gstapp-1.0 ${ARAVIS_LIB})		# gstreamer-0.10 gstbase-0.10 gstapp-0.10 


# transfer all headers to the include directory
foreach(include ${inferenceIncludes})
	message("-- Copying ${include}")
	configure_file(${include} ${PROJECT_INCLUDE_DIR} COPYONLY)
endforeach()


# copy network data
file(GLOB networkData ${PROJECT_SOURCE_DIR}/data/networks/* ${PROJECT_SOURCE_DIR}/data/images/*)

foreach(include ${networkData})
	message("-- Copying ${include}")
	file(COPY ${include} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
	#configure_file(${include} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COPYONLY)
endforeach()


# build samples & tools
#add_subdirectory(imagenet-console)
add_subdirectory(imagenet-camera)

#add_subdirectory(detectnet-console)
add_subdirectory(detectnet-camera)

#add_subdirectory(camera/gst-camera)
#add_subdirectory(camera/v4l2-console)
#add_subdirectory(camera/v4l2-display)

