// Author Dmytro Bobkov, dmytro.bobkov@tum.de
// Chair of Media Technology, Technical University of Munich, 2018-2019
// Parts of these were done while being at research visit to group of Bernd Girod IVMS at Stanford University

#pragma once

#include <iostream>
#include <iomanip>
#include <chrono>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <Eigen/Eigen>
#include <Eigen/Geometry>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include "json.hpp"

namespace po = boost::program_options;

struct Pose
{
    Eigen::Quaternionf orientation;
    Eigen::Vector3f location;
};

struct Panorama {
    Pose pose;
    std::string room;
};

struct PanoConfig {
    float shift_yaw_deg;
    bool invert_yaw;
};

struct RenderedImage {
    std::string filename_image;
    std::string filename_pano;
    Pose pose;
    int id = 0;
    int ind_yaw;

    float focal_length;
    int image_width;
    int image_height;
    std::string room;
};

/**
 *
 */
struct Configuration {
    //specify focal length of the final pinhole image
    double f = 350;

    //pixels are painted black for initialization
    int mat_height = 480;
    int mat_width = 640;

    int number_yaw_angles = 12;
    std::vector<float> pitch_angles = {-15, 0, 15};
    std::vector<float> yaw_angles;

    Eigen::Matrix3d calibration_matrix;
};

void
initConfiguration(  Configuration& config);

void
initConfigurationMultipleRates( const std::vector<int>& yaw_rates,
                                Configuration& config);

// converts degrees in radians
double
radians( const double degrees );

/**
 * calculates the rotation matrix given three euler angles
 * @param rotx rotation around X axis
 * @param roty rotation around Y axis
 * @param rotz rotation around Z axis
 * @return Rotation matrix
 */
Eigen::Matrix3d
eul2rotm( const double rotx,
          const double roty,
          const double rotz);

/**
 * function to compute the position of the pixel in the equirectangular image given the pixel positions in the output image with
 * corresponding orientation and intrinsic calibration matrix
 * @param output_img_x X position of the pixel in the image
 * @param output_img_y Y position of the pixel in the image
 * @param Rot - rotation matrix (3x3)
 * @param w1 - image width
 * @param h1 - image height
 * @param K  - intrinsic calibration matrix
 * @return pixel coordinates in the equirectangular image
 */
std::pair<double, double>
reprojection( const int output_img_x,
              const int output_img_y,
              const Eigen::Matrix3d& Rot,
              const int w1,
              const int h1,
              const Eigen::Matrix3d& K );


std::string
getNameImage(   const std::string& name_prefix,
                const double& yaw_deg,
                const double& pitch_deg );

bool
computePerspectiveProjectionImage( const Eigen::Matrix3d& Rot,
                                   const Eigen::Matrix3d& K,
                                   const int dst_width,
                                   const int dst_height,
                                   const cv::Mat& img_src,
                                   cv::Mat& perspectiveProjection );

void
processOnePano( const Configuration& config_rendering,
                const PanoConfig& pano_config,
                const cv::Mat& img_src,
                const std::string& output_folder,
                const std::string& name_prefix );

double
constrainAngle(double x);


double
constrainRad(double x);

std::string
getStringFloat( const float value,
                const int precision );

/**
 * Create JSON representation of the rendered image for file writing
 * @param rendered_image_one_view
 * @return
 */
nlohmann::json
renderedViewToJSON( const RenderedImage& rendered_image_one_view );

/**
 * Init object rendered image with corresponding yaw, pitch angles from the given panorama.
 * Used for Stanford dataset
 * @param yaw_deg - image yaw angle in degrees
 * @param pitch_deg - image pitch angle in degrees
 * @param pano object storing pose
 * @param name_prefix
 * @param name_basename
 * @return
 */
RenderedImage
getRenderedImage(   const float yaw_deg,
                    const float pitch_deg,
                    const Panorama& pano,
                    const std::string& name_prefix,
                    const std::string& name_basename );
