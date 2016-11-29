
cmake_minimum_required(VERSION 2.8)
project(seg-img-tool)




# Qt4 is used to load images (installed by ubuntu-desktop)
find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})


file(GLOB imgSources *.cpp)
file(GLOB imgIncludes *.h )

add_executable(seg-img-tool ${imgSources})
target_link_libraries(seg-img-tool Qt4::QtGui png)		# gstreamer-0.10 gstbase-0.10 gstapp-0.10 


file(GLOB scripts *.sh)
foreach(script ${scripts})
	message("-- Copying ${script}")
	configure_file(${script} ${PROJECT_BINARY_DIR} COPYONLY)
endforeach()

