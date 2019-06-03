// Author Dmytro Bobkov, dmytro.bobkov@tum.de
// Chair of Media Technology, Technical University of Munich, 2017-2018

#include <iostream>
#include <fstream>
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

namespace po = boost::program_options;
namespace fs = boost::filesystem;

#include "projection_helpers.h"
#include "io_helpers.h"

#include "rendering_tools.h"

#define IMG_WIDTH 4096
#define IMG_HEIGHT 2048

void
parseCommandLineOptions(int argc,
                        const char** argv,
                        Configuration& config_rendering,
                        std::string& in_pano_dir,
                        std::string& in_poses_dir,
                        std::string& output_folder,
                        bool& render_views_from_panoramas )
{
    po::options_description desc("Allowed options");
    desc.add_options()
            ("in_pano_dir", po::value<std::string>(), "Input pano directory")
            ("in_poses_dir", po::value<std::string>(), "Input poses directory")
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
}

int main( int argc, const char** argv )
{
    std::string in_pano_dir, output_folder, in_poses_dir;
    bool render_views_from_panoramas;
    Configuration config_rendering;

    parseCommandLineOptions(argc, argv, config_rendering,
                            in_pano_dir, in_poses_dir,
                            output_folder, render_views_from_panoramas);

    std::map<std::string, Panorama> pano_poses;
    parseInLocPanoramaPoses( in_poses_dir, pano_poses );

    PanoConfig pano_config;
    pano_config.shift_yaw_deg = 90;
    pano_config.invert_yaw = true;



    std::vector<std::string> pano_filenames;
    parsePanoramaFilenames( in_pano_dir, pano_filenames );

    assert(pano_filenames.size() == pano_poses.size());


    if( boost::filesystem::exists(output_folder)==false ) {
        boost::filesystem::create_directory(output_folder);
    }

    bool generate_multiple_yaw_rates = config_rendering.number_yaw_angles==-1;
    if( generate_multiple_yaw_rates ) {
        std::vector<int> yaw_numbers = {4, 8, 12, 16};
        initConfigurationMultipleRates(  yaw_numbers, config_rendering );
    }
    else {
        initConfiguration(  config_rendering );
    }
    // NOTE: caution we are using portrait dimensions because our queries are portrait!
    config_rendering.mat_width = 640;
    config_rendering.mat_height = 480;


    std::vector<RenderedImage> rendered_images;
    generateRenderedImages( config_rendering, pano_filenames, pano_poses, rendered_images );




    std::cout << "Finished rendered images " << rendered_images.size() << std::endl;

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
        if( img_src.empty() ) {//is image loaded?
            std::cerr << "Error: Could not load image!" << std::endl;
            continue;
        }

        // resize to standard size
        cv::Size standard_size(IMG_WIDTH, IMG_HEIGHT);
        cv::resize(img_src, img_src, standard_size);
        assert(img_src.rows() == IMG_HEIGHT);
        assert(img_src.cols() == IMG_WIDTH);

        processOnePano( config_rendering,
                        pano_config,
                        img_src,
                        output_folder,
                        name_prefix );

    }

    return 0;
}
