# Author Dmytro Bobkov
# Email dmytro.bobkov@tum.de
# Modified on 28.06.2018

cmake_minimum_required(VERSION 2.8)
project(view_rendering)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") #  -O3

set(OpenCV_DIR "$ENV{OpenCV_DIR}")

message("Found at ${OpenCV_DIR}")

find_package( OpenCV REQUIRED )

find_package(Boost 1.4 REQUIRED COMPONENTS
        system filesystem program_options graph regex)


find_package( Eigen3 REQUIRED ) #

message(WARNING "NOTE: Eigen directory parsed is $ENV{EIGEN3_INCLUDE_DIR}. If it is empty, try setting it to /usr/include/eigen3!")
INCLUDE_DIRECTORIES ( "$ENV{EIGEN3_INCLUDE_DIR}" ) # /usr/include/eigen3

set(INCLUDE_DIRS
        ${Boost_INCLUDE_DIRS})

set(LIBRARY_DIRS
        ${Boost_LIBRARY_DIRS})

set(LIBRARIES
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS})

include_directories(${INCLUDE_DIRS})
link_directories(${LIBRARY_DIRS})

set(HEADERS
        projection_helpers.h
        projection_helpers.cpp
        rendering_tools.h
        rendering_tools.cpp
        json.hpp
        io_helpers.h
        io_helpers.cpp
        )

##################
add_executable(render_views_from_pano_matterport render_views_from_pano_matterport.cpp ${HEADERS})
target_link_libraries(render_views_from_pano_matterport ${LIBRARIES} )

##################
add_executable(render_views_from_pano_inloc render_views_from_pano_inloc.cpp ${HEADERS})
target_link_libraries(render_views_from_pano_inloc ${LIBRARIES} )
