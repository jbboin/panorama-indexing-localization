// Author Dmytro Bobkov, dmytro.bobkov@tum.de
// Chair of Media Technology, Technical University of Munich, 2017-2018
#include "projection_helpers.h"

double
radians( const double degrees )
{
    return degrees*M_PI/180.;
}

void
initConfiguration(  Configuration& config)
{
    // create camera matrix K
    Eigen::Matrix3d K;
    K << config.f, 0, config.mat_width/2,
            0, config.f, config.mat_height/2,
            0,0,1;

    config.calibration_matrix = K;

    config.yaw_angles.resize(config.number_yaw_angles);
    float begin_angle_yaw = 0.f;
    float total_yaw = 360.f;
    float step_yaw = total_yaw / config.number_yaw_angles;
    for( size_t i=0; i<config.number_yaw_angles; i++ ) {
        config.yaw_angles[i] = begin_angle_yaw + (float)i * step_yaw;
    }
}

void
initConfigurationMultipleRates( const std::vector<int>& yaw_rates,
                                Configuration& config)
{
    // create camera matrix K
    Eigen::Matrix3d K;
    K << config.f, 0, config.mat_width/2,
            0, config.f, config.mat_height/2,
            0,0,1;

    config.calibration_matrix = K;

    // now sample yaw angles from multiple rates, and then merge accordingly

    float begin_angle_yaw = 0.f;
    float total_yaw = 360.f;

    std::set<float> yaw_angles;
    for( size_t yaw_ind=0; yaw_ind<yaw_rates.size(); yaw_ind++ ) {
        int number_yaw_angles = yaw_rates[yaw_ind];

        {
            float step_yaw = total_yaw / number_yaw_angles;
            for( size_t i=0; i<number_yaw_angles; i++ ) {
                float this_yaw = begin_angle_yaw + (float)i * step_yaw;
                yaw_angles.insert(this_yaw);
            }
        }
    }

    int number_yaw_angles_total = yaw_angles.size();

    config.number_yaw_angles = number_yaw_angles_total;
    config.yaw_angles.resize(config.number_yaw_angles);
    int cnt_yaw = 0;
    for( auto& m : yaw_angles ) {
        std::cout << "Putting yaw " << cnt_yaw << ":" << m << std::endl;
        config.yaw_angles[cnt_yaw] = m;
        cnt_yaw++;
    }


}

Eigen::Matrix3d
eul2rotm( const double rotx,
          const double roty,
          const double rotz)
{
    // Calculate rotation about x axis
    Eigen::Matrix3d R_x, R_y, R_z;
    R_x << 1,       0,              0,
            0,       cos(rotx),   -sin(rotx),
            0,       sin(rotx),   cos(rotx);
    // Calculate rotation about y axis
    R_y << cos(roty),    0,      sin(roty),
            0,               1,      0,
            -sin(roty),   0,      cos(roty);
    // Calculate rotation about z axis
    R_z  << cos(rotz),    -sin(rotz),      0,
            sin(rotz),    cos(rotz),       0,
            0,               0,                  1;
    // Combined rotation matrix
    Eigen::Matrix3d R = R_z * R_y * R_x;
    return R;
}


std::pair<double, double>
reprojection( const int output_img_x,
              const int output_img_y,
              const Eigen::Matrix3d& Rot,
              const int w1,
              const int h1,
              const Eigen::Matrix3d& K )
{
    //get 3D coordinates of pixel in img_interp (= final viewport)
    Eigen::Vector3d xyz(output_img_x, output_img_y, 1); //homogeneous coord.
    Eigen::Vector3d xyz_norm = xyz / xyz.norm(); //normalize
    Eigen::Matrix3d RK = Rot * K.inverse(); // R*K^-1
    Eigen::Vector3d ray3d = RK * xyz_norm; //2D-to-3D conversion

    // get 3d spherical coordinates
    double xp = ray3d(0);
    double yp = ray3d(1);
    double zp = ray3d(2);

    // inverse formula for spherical projection, reference Szeliski book "Computer Vision: Algorithms and Applications" p439.
    double theta = std::atan2(yp, sqrt(xp*xp + zp*zp));
    double phi = std::atan2(xp, zp);

    //get 2D point on equirectangular map
    // w1/pi and h1/pi = number of pixel per degree in horizontal and vertical direction
    double x_sphere = ((phi*w1)/M_PI+w1)/2; //need to shift by w1/2 because image coordinate system is in top left corner
    double y_sphere = (theta+ M_PI/2)*h1/M_PI;

    //return the point on the equirectangular picture
    return std::make_pair(x_sphere, y_sphere);
}



bool
computePerspectiveProjectionImage( const Eigen::Matrix3d& Rot,
                                   const Eigen::Matrix3d& K,
                                   const int dst_width,
                                   const int dst_height,
                                   const cv::Mat& img_src,
                                   cv::Mat& perspectiveProjection )
{
    //determine size of pano
    int src_height = img_src.rows;
    int src_width = img_src.cols;

    //loop over every pixel in output rectlinear image
    for( int output_img_v = 0; output_img_v < dst_height; ++output_img_v ) {
        for( int output_img_u = 0; output_img_u < dst_width; ++output_img_u ) {

            //determine corresponding position in the equirectangular panorama
            std::pair<double, double> current_pos = reprojection(output_img_u, output_img_v,
                                                                 Rot,
                                                                 src_width, src_height,
                                                                 K);

            //extract the x and y value of the position in the equirect. panorama
            int current_x = (int)std::floor(current_pos.first);
            int current_y = (int)std::floor(current_pos.second);

            // determine the nearest top left pixel for bilinear interpolation
            int top_left_x = current_x; //convert the subpixel value to a proper pixel value (top left pixel due to int() operator)
            int top_left_y = current_y;

            // this if statement added to mitigate the problem with some black pixels remaining after backprojection (by Dmytro)
            if( current_x<0 || top_left_x>src_width-1 || current_y<0 || top_left_y>src_height-1 ){
                if( current_x<0 ) {
                    current_x = 0;
                }

                if( top_left_x>src_width-1 ) {
                    top_left_x = src_width-1;
                }

                if( current_y<0 ) {
                    current_y = 0;
                }

                if( top_left_y>src_height-1 ) {
                    top_left_y = src_height-1;
                }
            }

            // if the current position exceeeds the panorama image size -- leave pixel black and skip to next iteration
            if( current_x<0 || top_left_x>src_width-1 || current_y<0 || top_left_y>src_height-1 ){
                std::cout << "Skipping position " << output_img_v << "x" << output_img_u << std::endl;
                continue;
            }

            // initialize weights for bilinear interpolation
            int dx = current_x - top_left_x;
            int dy = current_y - top_left_y;
            double wtl = (1.0-dx)*(1.0-dy); //weight top left
            double wtr = dx*(1.0-dy); // weight top right
            double wbl = (1.0-dx)*dy; // weight bottom left
            double wbr = dx*dy; // weight bottom right

            // determine subpixel value with bilinear interpolation
            cv::Vec3b bgr = wtl * img_src.at<cv::Vec3b>(top_left_y, top_left_x) +
                            wtr * img_src.at<cv::Vec3b>(top_left_y, top_left_x+1) +
                            wbl * img_src.at<cv::Vec3b>(top_left_y + 1, top_left_x) +
                            wbr * img_src.at<cv::Vec3b>(top_left_y + 1, top_left_x+1);

            // paint the pixel in the output image with the calculated value
            perspectiveProjection.at<cv::Vec3b>(cv::Point(output_img_u, output_img_v)) = bgr;
        }
    }
}

std::string
getStringFloat( const float value,
                const int precision )
{
    std::stringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    std::string s = stream.str();
    return s;
}

double
constrainRad(double x)
{
    x = fmod(x, 2*M_PI);
    return x;
}

double
constrainAngle(double x)
{
    x = fmod(x, 360);
    if (x < 0) {
        x += 360;
    }
    return x;
}

std::string
getNameImage(   const std::string& name_prefix,
                const double& yaw_deg,
                const double& pitch_deg )
{
    std::string name_view = name_prefix + "yaw_" + getStringFloat( yaw_deg, 1 );
    std::string pitch_string = getStringFloat( pitch_deg, 1 );
    name_view += "_pitch_" + pitch_string + ".png";
    return name_view;
}


void
processOnePano( const Configuration& config_rendering,
                const PanoConfig& pano_config,
                const cv::Mat& img_src,
                const std::string& output_folder,
                const std::string& name_prefix )
{
#pragma omp parallel
#pragma omp for
    for( size_t ind_yaw = 0; ind_yaw<config_rendering.yaw_angles.size(); ind_yaw++ ) {
        for( size_t ind_pitch = 0; ind_pitch<config_rendering.pitch_angles.size(); ind_pitch++ ) {

            const double yaw_ref = config_rendering.yaw_angles[ind_yaw];
            double yaw_deg = constrainAngle(yaw_ref);

            // This obscure line below is due to random rotation of panorama images
            // w.r.t. panorama orientation we observed in different datasets

            // e.g. for Stanford dataset, we have 90 degrees rotation
            // double yaw = -radians(yaw_deg-90);//- this works for stanford
            // while for Matterport it is 30 degrees
            // double yaw = radians(yaw_deg-30);//- this works for matterport
            double yaw = radians(yaw_deg - pano_config.shift_yaw_deg);
            if( pano_config.invert_yaw ) { // invert due to different conventions
                yaw = -yaw;
            }


            const double pitch_deg = config_rendering.pitch_angles[ind_pitch];
            double pitch = radians(pitch_deg);
            double roll = radians(0.0);

            Eigen::Matrix3d rot_view;
            rot_view = eul2rotm(-pitch, yaw, roll);
            Eigen::Matrix3d total_pose = rot_view;

            cv::Mat perspectiveProjection(config_rendering.mat_height, config_rendering.mat_width,
                                          CV_8UC3, cv::Scalar(0,0,0));

            auto t1 = std::chrono::high_resolution_clock::now();
            computePerspectiveProjectionImage( total_pose,
                                               config_rendering.calibration_matrix,
                                               config_rendering.mat_width,
                                               config_rendering.mat_height,
                                               img_src,
                                               perspectiveProjection );
            auto t2 = std::chrono::high_resolution_clock::now();

            std::string name_view = getNameImage(name_prefix, yaw_deg, pitch_deg);

            std::string fullname = output_folder + "/" + name_view;
            cv::imwrite(fullname.c_str(), perspectiveProjection);
            auto t3 = std::chrono::high_resolution_clock::now();

            int delta1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
            int delta2 = std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count();
            // to enable timing, uncomment line below
            // std::cout << "Computing perspective took " << delta1 << " and writing " <<
            // delta2 << " for [" << name_view << "]" << std::endl;
        }
    }
}

nlohmann::json
renderedViewToJSON( const RenderedImage& rendered_image_one_view )
{
    nlohmann::json one_view;
    Eigen::Quaternionf quat(rendered_image_one_view.pose.orientation);
    std::vector<float> orient = {quat.w(),
                                 quat.x(),
                                 quat.y(),
                                 quat.z()};

    for( size_t i=0; i<orient.size(); i++ ) {
        one_view["final_camera_rotation"][i] = orient[i];
    }

    std::vector<float> loc(rendered_image_one_view.pose.location.data(),
                           rendered_image_one_view.pose.location.data() + rendered_image_one_view.pose.location.rows() *
                                                                          rendered_image_one_view.pose.location.cols());

    one_view["camera_location"] = loc;
    one_view["room"] = rendered_image_one_view.room;
    one_view["filename"] = rendered_image_one_view.filename_image;
    one_view["source_pano_filename"] = rendered_image_one_view.filename_pano;
    one_view["id"] = rendered_image_one_view.id;
    one_view["ind_yaw"] = rendered_image_one_view.ind_yaw;

    return one_view;
}

RenderedImage
getRenderedImage(   const float yaw_deg,
                    const float pitch_deg,
                    const Panorama& pano,
                    const std::string& name_prefix,
                    const std::string& name_basename )
{
    Eigen::Quaternionf rot_view_quat;

    double yaw = radians(yaw_deg);//-
    double pitch = radians(pitch_deg);
    double roll = radians(0.0);
    Eigen::Vector3f euler_view(yaw, pitch, roll);

    Eigen::Matrix3d rot_view;
    rot_view = eul2rotm(-pitch, yaw, roll);
    rot_view_quat = rot_view.cast<float>();

    std::string name_view = getNameImage(name_prefix, yaw_deg, pitch_deg);
    //std::string fullname = output_folder + name_view;
    RenderedImage rendered_image;
    rendered_image.filename_image = name_view;
    rendered_image.pose.location = pano.pose.location;

    //yaw,roll,pitch // Y, Z, X axes
    const Eigen::Vector3f euler_pano = pano.pose.orientation.toRotationMatrix().eulerAngles(1, 2, 0);


    Eigen::Vector3f euler;
    for( size_t i=0; i<3; i++ ) {
        euler[i] = euler_pano[i] * 180. / M_PI;
    }

    float yaw_pano = euler_pano[0];
    float yaw_view = euler_view[0];

    float roll_pano = euler_pano[1];
    float roll_view = euler_view[1];

    float pitch_pano = euler_pano[2];
    float pitch_view = euler_view[2];

    Eigen::Quaternionf rot_total;
    rot_total = Eigen::AngleAxisf(yaw_pano+yaw_view+M_PI_2, Eigen::Vector3f::UnitZ()) *
                Eigen::AngleAxisf(roll_pano+roll_view, Eigen::Vector3f::UnitY()) *
                Eigen::AngleAxisf(pitch_pano+pitch_view, Eigen::Vector3f::UnitX());

    rendered_image.pose.orientation = rot_total;
    rendered_image.filename_pano = name_basename;
    rendered_image.room = pano.room;

    return rendered_image;
}
