// Author Dmytro Bobkov, dmytro.bobkov@tum.de
// Chair of Media Technology, Technical University of Munich, 2017-2018

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

// OpenCV
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// Eigen
#include <Eigen/Eigen>
#include <Eigen/Geometry>

// Boost
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

#include "projection_helpers.h"
#include "io_helpers.h"
#include "rendering_tools.h"


void
parseCommandLineOptions(int argc,
                        const char** argv,
                        Configuration& config_rendering,
                        std::string& in_pano_dir,
                        std::string& in_poses_dir,
                        std::string& in_room_segmentations,
                        std::string& output_folder,
                        bool& render_views_from_panoramas )
{
    po::options_description desc("Allowed options");
    desc.add_options()
            ("in_pano_dir", po::value<std::string>(), "Input pano directory")
            ("in_poses_dir", po::value<std::string>(), "Input poses directory")
            ("in_rooms_file", po::value<std::string>(), "Input poses directory")
            ("out_rgb_view_dir", po::value<std::string>(), "Outdirecotry")
            ("render_views_from_panoramas", po::bool_switch(&render_views_from_panoramas), "Optional argument for rendering")
            ("num_yaw_angles", po::value<int>(), "num_yaw_angles");;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("in_pano_dir")) {
        in_pano_dir = vm["in_pano_dir"].as<std::string>();
        std::cout << "In pano dir set to " << in_pano_dir << ".\n";
    }

    if (vm.count("in_poses_dir")) {
        in_poses_dir = vm["in_poses_dir"].as<std::string>();
        std::cout << "In poses dir set to " << in_poses_dir << ".\n";
    }

    if (vm.count("out_rgb_view_dir")) {
        output_folder = vm["out_rgb_view_dir"].as<std::string>();
        std::cout << "Out dir set to " << output_folder << ".\n";
    }

    if (vm.count("num_yaw_angles")) {
        config_rendering.number_yaw_angles = vm["num_yaw_angles"].as<int>();
        std::cout << "num_yaw_angles set to " << config_rendering.number_yaw_angles << ".\n";
    }

    if( vm.count("in_rooms_file")) {
        in_room_segmentations = vm["in_rooms_file"].as<std::string>();
        std::cout << "in_room_segmentations set to " << in_room_segmentations << ".\n";
    }
}

/**
 * This is for matterport3D dataset pano/pose folder
 * @param folder_with_poses
 * @param pano_infos
 */
void
parsePanoramasAsAverageOfSingleViews( const std::string& folder_with_view_poses,
                                      std::map<std::string, Panorama>& pano_infos )
{
    std::vector<std::string> files;



    // Note for (auto & p : fs::directory_iterator(folder_with_view_poses)) does not work for gcc 4.9

    for( auto& p : boost::make_iterator_range(fs::directory_iterator(folder_with_view_poses), {} )) {
        files.push_back(p.path().string());
    }

    std::map<std::string, Pose> view_to_pose;
    std::map<std::string, std::map< std::string, Pose > > pano_to_view_poses;

    for( std::string& txt_file : files ) {

        boost::filesystem::path p_i(txt_file);

        std::string extension = boost::filesystem::extension(p_i);
        if( extension != ".txt" ) {
            continue;
        }

        // split
        std::string delimiter = "_";
        std::string basename = boost::filesystem::basename(txt_file);
        std::string panorama_name = basename.substr(0, basename.find(delimiter));

        Eigen::MatrixXf rt_matrix4x4 = parseEigenMatrix( txt_file );

        // initialize the pose and rest correctly
        Pose view_pose;

        Eigen::Matrix3f rot_pose = rt_matrix4x4.topLeftCorner(3,3);
        Eigen::Vector3f t = rt_matrix4x4.topRightCorner(3, 1);
        Eigen::Quaternionf quat_rot(rot_pose);

        view_pose.location = t;
        view_pose.orientation = rot_pose;

        view_to_pose.insert({basename, view_pose});

        if( pano_to_view_poses.count( panorama_name) == 0 ) {
            std::map<std::string, Pose> views;
            views.insert({basename, view_pose});
            pano_to_view_poses.insert({panorama_name, views});
        }
        else {
            pano_to_view_poses[panorama_name].insert({basename, view_pose});
        }
    }

    // now we have collected all views for each panorama, so we can actually
    // get the average as our location and for pose we would just get the first view in the middle plane
    for( auto& pano_views : pano_to_view_poses ) {
        std::string pano_name = pano_views.first;

        std::map<std::string, Pose> views = pano_views.second;

        // set pose of the panorama as the average of all camera locations at this panorama index
        Eigen::Vector3f location_average(0,0,0);
        for( auto& view_this : views ) {
            //std::cout << "This " << (view_this.first) << std::endl;
            location_average += view_this.second.location;
        }
        location_average /= views.size();

        std::string middle_view_name = pano_name + "_pose_1_0";//get first horizontal view

        Eigen::Quaternionf pano_orient;
        pano_orient = views.at(middle_view_name).orientation;

        Panorama panorama;
        panorama.pose.location = location_average;
        panorama.pose.orientation = pano_orient;

        std::string pano_correct_name = pano_name; // "camera_" +

        pano_infos.insert({pano_correct_name, panorama});
    }

    std::cout << "Finished view parsing" << std::endl;
}

RenderedImage
generateRenderedImage( const std::string& name_prefix,
                       const std::string& name_basename,
                       const Panorama& pano,
                       const float yaw_deg,
                       const float pitch_deg,
                       const std::string& output_folder)
{
    // get panorama orientation
    const Eigen::Quaternionf& pano_orient = pano.pose.orientation;

    RenderedImage rendered_image;

    double yaw = radians(yaw_deg);//-
    double pitch = radians(pitch_deg);
    if( pitch!=0 ) {
        pitch *= -1;
    }
    double roll = radians(0.0);

    std::string name_view = getNameImage(name_prefix, yaw_deg, pitch_deg);
    std::string fullname = output_folder + name_view;
    rendered_image.filename_image = name_view;

    //std::cout << "Input quat " << pano_orient.w() << "," << pano_orient.vec().transpose() << std::endl;
    Eigen::Vector3f euler_pano = pano_orient.toRotationMatrix().eulerAngles(2, 1, 0); //yaw, roll, pitch
    Eigen::Vector3f euler_view(-yaw, roll, pitch);

    Eigen::Vector3f euler_total = euler_pano + euler_view;
    for( size_t i=0; i<3; i++ ) {
        euler_total(i) = constrainRad(euler_total(i));
    }

    Eigen::Quaternionf rot_total;
    rot_total = Eigen::AngleAxisf(euler_total(0), Eigen::Vector3f::UnitZ()) *
                Eigen::AngleAxisf(euler_total(1), Eigen::Vector3f::UnitY()) *
                Eigen::AngleAxisf(euler_total(2), Eigen::Vector3f::UnitX());

    //std::cout << "View " << name_view << " has " << rot_total.w() << "," << rot_total.vec().transpose() << std::endl;

    rendered_image.pose.orientation = rot_total;
    rendered_image.filename_pano = name_basename;
    rendered_image.room = pano.room;
    rendered_image.pose.location = pano.pose.location;

    return rendered_image;
}




/*
 Example how to run
 --in_pano_dir /media/dima/Data/Datasets/Matterport3D/v1/scans/2n8kARJN3HM/pano/
 --in_pose_dir /media/dima/Data/Datasets/Matterport3D/v1/scans/2n8kARJN3HM/matterport_camera_poses/
 --out_rgb_view_dir /media/dima/Data/Datasets/Matterport3D/v1/scans/2n8kARJN3HM/rendered_10/
 --num_yaw_angles 10
 */

int main( int argc, const char** argv )
{
    std::string in_pano_dir, output_folder, in_poses_dir, in_room_segmentations;
    Configuration config_rendering;
    //config_rendering.pitch_angles = {0};
    bool render_views_from_panoramas;

    parseCommandLineOptions(argc, argv,
            config_rendering,
            in_pano_dir,
            in_poses_dir,
            in_room_segmentations,
            output_folder,
            render_views_from_panoramas);

    PanoConfig pano_config;
    pano_config.shift_yaw_deg = 30;
    pano_config.invert_yaw = false;


    std::map<std::string, Panorama> panos;
    parsePanoramasAsAverageOfSingleViews( in_poses_dir, panos );


    std::map<std::string, std::string> panorama_to_room;
    parsePanoramaToRooms(in_room_segmentations, panorama_to_room);
    {
        for( auto& pano : panos ) {
            std::string room = panorama_to_room.at(pano.first);
            pano.second.room = room;
        }

        // now augment filename with camera_X_roomlabel_roomid
        std::map<std::string, Panorama> panos_correct;
        for( auto& pano : panos ) {
            std::string pano_correct = "camera_" + pano.first + "_" + pano.second.room;
            //std::cout << "Correct " << pano_correct << std::endl;
            panos_correct.insert({pano_correct, pano.second});
        }

        panos = panos_correct;

    }


    if( boost::filesystem::exists(output_folder)==false ) {
        boost::filesystem::create_directory(output_folder);
    }


    bool generate_multiple_yaw_rates = config_rendering.number_yaw_angles==-1;
    if( generate_multiple_yaw_rates ) {
        std::vector<int> yaw_numbers = {8, 10, 12, 16};
        initConfigurationMultipleRates(  yaw_numbers, config_rendering );
    }
    else {
        initConfiguration(  config_rendering );

    }




    std::vector<std::string> pano_filenames;
    if( boost::filesystem::exists(in_pano_dir) ) {
        boost::filesystem::path p(in_pano_dir);
        for( auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(p), {})) {
            std::string extension =  boost::filesystem::extension(entry);
            if( extension == ".png" || extension == ".jpg" ) {
                std::string name_file = entry.path().string();
                pano_filenames.push_back(name_file);
            }
            else {
                std::cout << "Not png or jpg" << entry.path().string() << std::endl;
            }
        }
    }
    else {
        std::cout << "Folder " << in_pano_dir << " does not exist" << std::endl;
        exit(-1);
    }


    std::vector<RenderedImage> rendered_images;
    int global_view_id = 0;
    for( size_t ind_pano_raw=0; ind_pano_raw<pano_filenames.size(); ind_pano_raw++ ) {
        std::string name_image = pano_filenames[ind_pano_raw];
        // get index from the filename correctly
        boost::filesystem::path p_i(name_image);
        std::string pano_name_file = p_i.stem().string(); // without extension

        std::vector<std::string> strs;

        boost::split(strs, pano_name_file, boost::is_any_of("_"));
        //std::cout << "Split of " << pano_name_file << " into " << strs.size() << ":" << strs[1] << std::endl;
        assert(strs.size()==4);//camera_panoname_roomname_id
        std::string pano_name_stripped = pano_name_file;//strs[1];

        // remove

        std::string name_basename = p_i.filename().string();
        std::string name_prefix = pano_name_stripped + '_';

        if( panos.count(pano_name_stripped)==0 ) {
            std::cout << "Looking at panorama " << pano_name_stripped << ", but not present. Skipping!" << std::endl;
            continue;
        }

        const Panorama& pano = panos.at(pano_name_stripped);

        for (size_t ind_yaw = 0; ind_yaw < config_rendering.yaw_angles.size(); ind_yaw++) {
            for (size_t ind_pitch = 0; ind_pitch < config_rendering.pitch_angles.size(); ind_pitch++) {
                const double yaw_ref = config_rendering.yaw_angles[ind_yaw];
                double yaw_deg = constrainAngle(yaw_ref);
                const double pitch_deg = config_rendering.pitch_angles[ind_pitch];

                RenderedImage rendered_image = generateRenderedImage( name_prefix,
                                                                      name_basename,
                                                                      pano,
                                                                      yaw_deg,
                                                                      pitch_deg,
                                                                      output_folder);
                rendered_image.id = global_view_id;
                rendered_image.ind_yaw = ind_yaw;
                global_view_id++;

                rendered_images.push_back(rendered_image);
            }
        }
    }

    std::cout << "Finished computing rendered images " << rendered_images.size() << std::endl;

    // now write to JSON all data
    std::string json_output_all_rendered_view = output_folder + "/view_poses.json";
    writeViewsAndConfigToJSON( config_rendering, rendered_images, json_output_all_rendered_view );



    if( !render_views_from_panoramas ) {
        return 0;
    }


    // now iterate over all panos
    for( size_t ind_pano_raw=0; ind_pano_raw<pano_filenames.size(); ind_pano_raw++ ) {
        std::cout << "Rendering from pano " << ind_pano_raw << "/" << pano_filenames.size() << std::endl;

        std::string name_image = pano_filenames[ind_pano_raw];

        // get index from the filename correctly
        boost::filesystem::path p_i(name_image);
        std::string name_file = p_i.stem().string();
        std::string name_prefix = name_file + '_';

        cv::Mat img_src = cv::imread(name_image.c_str(), CV_LOAD_IMAGE_COLOR);
        if( img_src.empty() ) {
            std::cout << "Error: Could not load image!" << std::endl;
            continue;
        }

        processOnePano( config_rendering,
                        pano_config,
                        img_src,
                        output_folder,
                        name_prefix );
    }

    return 0;
}
