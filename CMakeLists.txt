cmake_minimum_required(VERSION 2.8)
project( HOG )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( HOGCascade.hpp )
add_executable( HOG HOG.cpp HOGCascade.cpp )
target_link_libraries( HOG ${OpenCV_LIBS} )

