// Author Dmytro Bobkov, dmytro.bobkov@tum.de
// Chair of Media Technology, Technical University of Munich, 2017-2018

#include "io_helpers.h"


void
parseCameraPoses( const std::string& json_file,
                  std::map<int, RenderedImage>& rendered_views )
{


    boost::filesystem::path p_i(json_file);

    std::string extension = boost::filesystem::extension(p_i);
    if( extension != ".json" ) {
        exit(-1);
    }

    std::cout << "Parsing camera poses from " << json_file << std::endl;

    std::ifstream json_input(json_file);
    nlohmann::json json_total;
    json_input >> json_total;

    nlohmann::json images_json = json_total["images"];
    for (nlohmann::json::iterator it = images_json.begin(); it != images_json.end(); ++it) {

        auto rendered_image_json = it.value();

        RenderedImage rendered_image_one_view;

        try {
            std::vector<float> location_vec = rendered_image_json["camera_location"];
            std::vector<float> rotation = rendered_image_json["final_camera_rotation"];

            rendered_image_one_view.pose.location = Eigen::Vector3f(location_vec.data());

            Eigen::Quaternionf quat_input = Eigen::Quaternionf(rotation[0], rotation[1], rotation[2], rotation[3]);
            // w, x, y, z // rotation.data()
            Eigen::Vector3f euler_total = quat_input.toRotationMatrix().eulerAngles(0, 1, 2);
            //yaw,roll,pitch

            Eigen::Quaternionf quat_correct;
            quat_correct =  Eigen::AngleAxisf(euler_total[0], Eigen::Vector3f::UnitX()) *
                            Eigen::AngleAxisf(euler_total[1], Eigen::Vector3f::UnitY()) *
                            Eigen::AngleAxisf(euler_total[2], Eigen::Vector3f::UnitZ());

            rendered_image_one_view.pose.orientation = quat_correct;
            rendered_image_one_view.filename_image = rendered_image_json["filename"];
            rendered_image_one_view.filename_pano = rendered_image_json["source_pano_filename"];

            rendered_image_one_view.id = rendered_image_json["id"];
            rendered_image_one_view.ind_yaw = rendered_image_json["ind_yaw"];
            rendered_image_one_view.room = rendered_image_json["room"];

            rendered_views[rendered_image_one_view.id] = rendered_image_one_view;
        }
        catch ( std::domain_error &e )
        {
            std::cout << "Skipping image " << rendered_image_json["filename"] << "/" << images_json.size()
            << ", because values are missing " << std::endl;
        }
    }

    std::cout << "Finished parsing from JSON " << json_file << std::endl;
}

double
interpolate( const double val, const double y0, const double x0, const double y1, const double x1 )
{
    return (val-x0)*(y1-y0)/(x1-x0) + y0;
}

double
base( const double val )
{
    if ( val <= -0.75 ) return 0;
    else if ( val <= -0.25 ) return interpolate( val, 0.0, -0.75, 1.0, -0.25 );
    else if ( val <= 0.25 ) return 1.0;
    else if ( val <= 0.75 ) return interpolate( val, 1.0, 0.25, 0.0, 0.75 );
    else return 0.0;
}

double
red( const double gray )
{
    return base( gray - 0.5 );
}

double
green( const double gray )
{
    return base( gray );
}

double
blue( const double gray )
{
    return base( gray + 0.5 );
}


void
convertToColormapJet( const cv::Mat &input,
                      const float maxValue,
                      cv::Mat &output )
{
    assert(input.channels()==1);
    cv::Mat temp;
    if( input.type() != CV_32FC1 ) {
        input.convertTo(temp, CV_32FC1);
    }
    else {
        temp = input.clone();
    }
    // save also colormaped
    output.create(temp.rows, temp.cols, CV_32FC3);
    output.setTo(cv::Vec3f(255, 255, 255));//set background to white
    for( size_t i=0; i<temp.rows; i++ ) {
        for( size_t j=0; j<temp.cols; j++ ) {
            double value = temp.at<float>(i, j) * 1./maxValue;
            assert(value<=1.0);
            if( value>0.0 ) {
                double scaledValue = 2*value - 1;
                cv::Vec3f px;
                px[0] = blue(scaledValue)*255; // note the order BGR!
                px[1] = green(scaledValue)*255;
                px[2] = red(scaledValue)*255;
                output.at<cv::Vec3f>(i, j) = px;
            }
        }
    }
}

Eigen::MatrixXf
parseEigenMatrix( const std::string& txt_file )
{
    std::ifstream fin (txt_file.c_str());
    int nrows = 4, ncols = 4;
    Eigen::MatrixXf rt_matrix4x4 = Eigen::MatrixXf::Zero(nrows, ncols);
    if (fin.is_open()) {
        for (int row = 0; row < nrows; row++) {
            for (int col = 0; col < ncols; col++) {
                float item = 0.0;
                fin >> item;
                rt_matrix4x4(row, col) = item;
            }
        }
        fin.close();
    }
    return rt_matrix4x4;
}

void
writeViewsAndConfigToJSON( const Configuration& config_rendering,
                           const std::vector<RenderedImage>& rendered_images,
                           const std::string& json_output_all_rendered_view )
{
    std::ofstream o(json_output_all_rendered_view);
    nlohmann::json out_data;
    nlohmann::json out_data_images;

    for( size_t index_view=0; index_view<rendered_images.size(); index_view++ ) {
        RenderedImage rendered_image_one_view = rendered_images[index_view];
        nlohmann::json one_view = renderedViewToJSON( rendered_image_one_view );
        out_data_images[index_view] = one_view;
    }
    out_data["images"] = out_data_images;

    std::vector<float> calib_vec(config_rendering.calibration_matrix.transpose().data(),
            config_rendering.calibration_matrix.data() + 9);

    out_data["image_width"] = config_rendering.mat_width;
    out_data["image_height"] = config_rendering.mat_height;
    out_data["calibration_matrix"] = calib_vec;

    o << std::setw(4) << out_data << std::endl;
    o.close();
}

void
parsePanoramaToRooms( const std::string& in_room_segmentations,
                      std::map<std::string, std::string>& panorama_to_room )
{

    std::ifstream infile (in_room_segmentations.c_str());
    std::cout << "Parsing " << in_room_segmentations << std::endl;

    if (infile.is_open()) {
        std::string line;
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            std::string id, panorama, room_id, room_label;

            if (!(iss >> id >> panorama >> room_id >> room_label)) {
                break;
            } // error

            std::string label_total = room_label + "_" + room_id;
            panorama_to_room.insert({panorama, label_total});
        }
        infile.close();
    }
}
