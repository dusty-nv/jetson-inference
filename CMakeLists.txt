
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


# setup build flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-write-strings -Wno-deprecated-declarations")	# -std=c++14 
set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")
set(BUILD_INTERACTIVE "YES" CACHE BOOL "If NO, will download/install the default DNN models without prompting the user, and skip installation of PyTorch.")
set(BUILD_EXPERIMENTAL "NO" CACHE BOOL "If YES, will enable support for experimental DNNs, examples, and plugins")


# copy configuration tools to build dir
#file(COPY "tools/download-models.sh" DESTINATION ${PROJECT_BINARY_DIR})
#file(COPY "tools/download-models.rc" DESTINATION ${PROJECT_BINARY_DIR})
file(COPY "tools/install-pytorch.sh" DESTINATION ${PROJECT_BINARY_DIR})
file(COPY "tools/install-pytorch.rc" DESTINATION ${PROJECT_BINARY_DIR})


# detect distro version
find_program(LSB_RELEASE_EXEC lsb_release)

execute_process(COMMAND "${LSB_RELEASE_EXEC}" --short --id OUTPUT_VARIABLE LSB_RELEASE_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "${LSB_RELEASE_EXEC}" --short --release OUTPUT_VARIABLE LSB_RELEASE_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "${LSB_RELEASE_EXEC}" --short --codename OUTPUT_VARIABLE LSB_RELEASE_CODENAME OUTPUT_STRIP_TRAILING_WHITESPACE)

message("-- distro ID:       ${LSB_RELEASE_ID}")
message("-- distro version:  ${LSB_RELEASE_VERSION}")
message("-- distro codename: ${LSB_RELEASE_CODENAME}")


# if this is the first time running cmake, perform pre-build dependency install script (or if the user manually triggers re-building the dependencies)
if( ${BUILD_DEPS} )
	message("-- Launching pre-build dependency installer script...")
	message("-- Build interactive:  ${BUILD_INTERACTIVE}")

	execute_process(COMMAND sh ../CMakePreBuild.sh ${BUILD_INTERACTIVE}
				WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
				RESULT_VARIABLE PREBUILD_SCRIPT_RESULT)

	set(BUILD_DEPS "NO" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed." FORCE)
	message("-- Finished installing dependencies")
endif()


# setup CUDA
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/utils/cuda" )
find_package(CUDA)
message("-- CUDA version: ${CUDA_VERSION}")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
	message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_53 SM_62")
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_53,code=sm_53 -gencode arch=compute_62,code=sm_62)

	if(CUDA_VERSION_MAJOR GREATER 9)
		message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_72")
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_72,code=sm_72)
	endif()

	if(CUDA_VERSION_MAJOR GREATER 10)
		message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_87")
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_87,code=sm_87)
	endif()
endif()

# OpenCV used for findHomography() and decomposeHomography()
# OpenCV version >= 3.0.0 required for decomposeHomography()
find_package(OpenCV COMPONENTS core calib3d)

if( NOT OpenCV_FOUND )
	message("-- didn't find OpenCV on system, disabling OpenCV")
else()
	message("-- OpenCV version:  " ${OpenCV_VERSION})

	if( ${OpenCV_VERSION_MAJOR} LESS 3 )
		message("-- OpenCV version less than 3.0, disabling OpenCV")
	else()
		message("-- OpenCV version >= 3.0.0, enabling OpenCV")
		set(HAS_OPENCV 1)
		add_definitions(-DHAS_OPENCV)
	endif()	
endif()
	
# check for VPI (TODO: VPI 1.0 support for JetPack 4.x)
find_package(VPI 2.0)

if( NOT VPI_FOUND )
	message("-- didn't find VPI on system, disabling VPI")
else()
	message("-- VPI version:  " ${VPI_VERSION})
	set(HAS_VPI 1)
	add_definitions(-DHAS_VPI)
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

# build C/C++ library
include_directories(${PROJECT_INCLUDE_DIR} ${PROJECT_INCLUDE_DIR}/jetson-inference ${PROJECT_INCLUDE_DIR}/jetson-utils)
include_directories(/usr/include/gstreamer-1.0 /usr/lib/${CMAKE_SYSTEM_PROCESSOR}/gstreamer-1.0/include /usr/include/glib-2.0 /usr/include/libxml2 /usr/lib/${CMAKE_SYSTEM_PROCESSOR}/glib-2.0/include/)

file(GLOB inferenceSources c/*.cpp c/*.cu c/calibration/*.cpp c/tracking/*.cpp)
file(GLOB inferenceIncludes c/*.h c/*.cuh c/calibration/*.h c/tracking/*.h)

if(BUILD_EXPERIMENTAL)
	message("-- BUILD_EXPERIMENTAL enabled")

	file(GLOB experimentalSources c/experimental/*.cpp c/experimental/*.cu)
	file(GLOB experimentalIncludes c/experimental/*.h c/experimental/*.cuh)
	file(GLOB_RECURSE pluginSources c/plugins/*.cpp c/plugins/*.cu)

	list(APPEND inferenceSources ${experimentalSources})
	list(APPEND inferenceIncludes ${experimentalIncludes})
else()
	message("-- BUILD_EXPERIMENTAL disabled")
	file(GLOB pluginSources c/plugins/*.cpp c/plugins/*.cu c/plugins/pose/trt_pose/parse/*.cpp)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
	link_directories(/usr/lib/aarch64-linux-gnu/tegra)
endif()
	
cuda_add_library(jetson-inference SHARED ${inferenceSources} ${pluginSources})
target_include_directories(jetson-inference PRIVATE ${PROJECT_SOURCE_DIR}/c/plugins/pose/trt_pose/parse)

# transfer all headers to the include directory
file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR}/jetson-inference)

foreach(include ${inferenceIncludes})
	message("-- Copying ${include}")
	configure_file(${include} ${PROJECT_INCLUDE_DIR}/jetson-inference COPYONLY)
endforeach()

#if(BUILD_EXPERIMENTAL)

#endif()

# create symbolic link for network and image data
execute_process( COMMAND "${CMAKE_COMMAND}" "-E" "create_symlink" "${PROJECT_SOURCE_DIR}/data/networks" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/networks" )
execute_process( COMMAND "${CMAKE_COMMAND}" "-E" "create_symlink" "${PROJECT_SOURCE_DIR}/data/images" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/images" )

  
# copy image data (EDIT: these are now symlinked above)
#file(GLOB imageData ${PROJECT_SOURCE_DIR}/data/images/*)

#foreach(image ${imageData})
#	message("-- Copying ${image}")
#	file(COPY ${image} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
#	#configure_file(${include} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COPYONLY)
#endforeach()


# build subdirectories
add_subdirectory(docs)
add_subdirectory(examples)
add_subdirectory(tools)
add_subdirectory(utils)
add_subdirectory(python)


# set linker options
target_link_libraries(jetson-inference jetson-utils nvinfer nvinfer_plugin nvcaffe_parser)

if(CUDA_VERSION_MAJOR GREATER 9)
	target_link_libraries(jetson-inference nvonnxparser)
endif()

if(HAS_OPENCV) 
   message("-- linking jetson-inference with OpenCV " ${OpenCV_VERSION})
   target_link_libraries(jetson-inference opencv_core opencv_calib3d)
endif()

if(HAS_VPI) 
   message("-- linking jetson-inference with VPI " ${VPI_VERSION})
   target_link_libraries(jetson-inference vpi)
endif()

# install includes
foreach(include ${inferenceIncludes})
    install(FILES "${include}" DESTINATION include/jetson-inference)
endforeach()

# install symlink to networks and images
install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/data/networks ${CMAKE_INSTALL_PREFIX}/bin/networks )" )
install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/data/images ${CMAKE_INSTALL_PREFIX}/bin/images )" )

# install the shared library
install(TARGETS jetson-inference DESTINATION lib EXPORT jetson-inferenceConfig)

# install the cmake project, for importing
install(EXPORT jetson-inferenceConfig DESTINATION share/jetson-inference/cmake)

# run ldconfig after installing
install(CODE "execute_process( COMMAND ldconfig )")

