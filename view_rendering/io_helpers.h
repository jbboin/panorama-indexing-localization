// Author Dmytro Bobkov, dmytro.bobkov@tum.de
// Chair of Media Technology, Technical University of Munich, 2018-2019
// Parts of these were done while being at research visit to group of Bernd Girod IVMS at Stanford University

#pragma once

#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <Eigen/Eigen>
#include <Eigen/Geometry>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include "projection_helpers.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;

double interpolate( const double val,
                    const double y0,
                    const double x0,
                    const double y1,
                    const double x1 );

double base( const double val );
double red( const double gray );
double green( const double gray );
double blue( const double gray );

void
parseCameraPoses( const std::string& json_file,
                  std::map<int, RenderedImage>& rendered_views );

Eigen::MatrixXf
parseEigenMatrix( const std::string& txt_file );

void
writeViewsAndConfigToJSON( const Configuration& config_rendering,
                           const std::vector<RenderedImage>& rendered_images,
                           const std::string& json_output_all_rendered_view );

void
parsePanoramaToRooms( const std::string& in_room_segmentations,
                      std::map<std::string, std::string>& panorama_to_room );