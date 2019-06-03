// Author Dmytro Bobkov, dmytro.bobkov@tum.de
// Chair of Media Technology, Technical University of Munich, 2017-2018

#include "rendering_tools.h"

#include <Eigen/Eigen>
#include <Eigen/Geometry>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

#include "projection_helpers.h"
#include "io_helpers.h"


void
generateRenderedImages( const Configuration& config_rendering,
                        const std::vector<std::string>& pano_filenames,
                        const std::map<std::string, Panorama>& pano_poses,
                        std::vector<RenderedImage>& rendered_images )
{
    int global_view_id = 0;
    for( size_t ind_pano_raw=0; ind_pano_raw<pano_filenames.size(); ind_pano_raw++ ) {
        std::string name_image = pano_filenames[ind_pano_raw];
        // get index from the filename correctly
        boost::filesystem::path p_i(name_image);
        std::string pano_name_file = p_i.stem().string(); // without extension
        std::string name_basename = p_i.filename().string();
        std::string name_prefix = pano_name_file + '_';

        // pano file to json pose (remove 3 symbols and add pose
        std::string name_json_pano_pose = pano_name_file.substr(0, pano_name_file.length() - 3) + "pose";

        if( pano_poses.count(name_json_pano_pose) == 0) {
            name_json_pano_pose = pano_name_file;
        }
        std::cout << "Pose filename " << name_json_pano_pose << " from " << pano_name_file << std::endl;

        Panorama pano = pano_poses.at(name_json_pano_pose);
        Pose pose = pano.pose;


        for (size_t ind_yaw = 0; ind_yaw < config_rendering.yaw_angles.size(); ind_yaw++) {
            for (size_t ind_pitch = 0; ind_pitch < config_rendering.pitch_angles.size(); ind_pitch++) {

                // based on pitch and roll, compute quaternion
                const double yaw_ref = config_rendering.yaw_angles[ind_yaw];
                double yaw_deg = constrainAngle(yaw_ref);
                const double pitch_deg = config_rendering.pitch_angles[ind_pitch];

                RenderedImage rendered_image = getRenderedImage( yaw_deg, pitch_deg,
                                                                 pano, name_prefix, name_basename );
                rendered_image.id = global_view_id;
                rendered_image.ind_yaw = ind_yaw;
                global_view_id++;

                rendered_images.push_back(rendered_image);
            }
        }

    }
}

void
parsePanoramaFilenames( const std::string& in_pano_dir,
                        std::vector<std::string>& pano_filenames )
{
    if( boost::filesystem::exists(in_pano_dir) ) {
        boost::filesystem::path p(in_pano_dir);
        for( auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(p), {})) {
            std::string extension =  boost::filesystem::extension(entry);
            if( extension == ".png" ) {
                std::string name_file = entry.path().string();
                pano_filenames.push_back(name_file);
            }
            else {
                std::cout << "Not jpg" << entry.path().string() << std::endl;
            }
        }
    }
    else {
        std::cout << "Folder " << in_pano_dir << " does not exist" << std::endl;
        exit(-1);
    }
}


void
parseInLocPanoramaPoses( const std::string& folder_with_poses,
                         std::map<std::string, Panorama>& pano_infos )
{
    std::vector<std::string> files;
    //for (auto & p : fs::directory_iterator(folder_with_poses))
    for( const auto& p : boost::make_iterator_range(fs::directory_iterator(folder_with_poses), {}))
    {
        //std::cout << p << std::endl;
        files.push_back(p.path().string());
    }

    for( std::string& json_file : files ) {


        boost::filesystem::path p_i(json_file);

        std::string extension = boost::filesystem::extension(p_i);
        if( extension != ".json" ) {
            continue;
        }

        std::string basename_file = p_i.stem().string();

        std::ifstream json_input(json_file);
        nlohmann::json json_pose;
        json_input >> json_pose;

        nlohmann::json json_location = json_pose["camera_location"];
        Eigen::Vector3f t_correct;
        t_correct << json_location[0], json_location[1], json_location[2];

        std::vector<float> rot_vec = json_pose["final_camera_rotation"];

        Eigen::Quaternionf quat_rot(rot_vec.data());

        Panorama pano;
        pano.pose.location = t_correct;
        pano.pose.orientation = quat_rot;
        pano.room = json_pose["room"];

        std::cout << "Name: " << basename_file << std::endl;
        pano_infos.insert({basename_file, pano});
    }
    std::cout << "Finished parsing panorama poses, total " << pano_infos.size() << std::endl;

}


/**
 * This is for stanford dataset pano/pose folder
 * @param folder_with_poses
 * @param pano_infos
 */
void
parsePanoramaPoses( const std::string& folder_with_poses,
                    std::map<std::string, Panorama>& pano_infos )
{
    std::vector<std::string> files;
    //for (auto & p : fs::directory_iterator(folder_with_poses))
    for( const auto& p : boost::make_iterator_range(fs::directory_iterator(folder_with_poses), {}))
    {
        //std::cout << p << std::endl;
        files.push_back(p.path().string());
    }

    for( std::string& json_file : files ) {


        boost::filesystem::path p_i(json_file);

        std::string extension = boost::filesystem::extension(p_i);
        if( extension != ".json" ) {
            continue;
        }

        std::string basename_file = p_i.stem().string();

        std::ifstream json_input(json_file);
        nlohmann::json json_pose;
        json_input >> json_pose;

        nlohmann::json json_location = json_pose["camera_location"];
        Eigen::Vector3f location;
        location << json_location[0], json_location[1], json_location[2];

        nlohmann::json rt_matrix = json_pose["camera_rt_matrix"];
        float rt00 = rt_matrix[0][0];
        Eigen::Matrix3f rot_pose;
        rot_pose << rt_matrix[0][0], rt_matrix[0][1], rt_matrix[0][2],
                rt_matrix[1][0], rt_matrix[1][1], rt_matrix[1][2],
                rt_matrix[2][0], rt_matrix[2][1], rt_matrix[2][2];
        Eigen::Vector3f t(rt_matrix[0][3], rt_matrix[1][3], rt_matrix[2][3]);
        Eigen::Vector3f t_correct = -rot_pose.transpose() * t;


        std::vector<float> rot_vec = json_pose["final_camera_rotation"];
        float yaw = rot_vec[0];
        float pitch = rot_vec[1];
        float roll = rot_vec[2];
        Eigen::Quaternionf q =  Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitZ()) *
                                Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
                                Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitX());
        Eigen::Quaternionf quat_rot(rot_pose);

        Panorama pano;
        pano.pose.location = t_correct;
        pano.pose.orientation = quat_rot;
        pano.room = json_pose["room"];

        std::cout << "Name: " << basename_file << std::endl;
        pano_infos.insert({basename_file, pano});
    }
    std::cout << "Finished parsing panorama poses, total " << pano_infos.size() << std::endl;
}