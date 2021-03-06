// Author Dmytro Bobkov, dmytro.bobkov@tum.de
// Chair of Media Technology, Technical University of Munich, 2018-2019
// Parts of these were done while being at research visit to group of Bernd Girod IVMS at Stanford University

#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>

#include "projection_helpers.h"
#include "io_helpers.h"



void
generateRenderedImages( const Configuration& config_rendering,
                        const std::vector<std::string>& pano_filenames,
                        const std::map<std::string, Panorama>& pano_poses,
                        std::vector<RenderedImage>& rendered_images );

void
parsePanoramaFilenames( const std::string& in_pano_dir,
                        std::vector<std::string>& pano_filenames );

/**
 * This is for dataset pano/pose folder
 * @param folder_with_poses
 * @param pano_infos
 */
void
parsePanoramaPoses( const std::string& folder_with_poses,
                    std::map<std::string, Panorama>& pano_infos );


void
parseInLocPanoramaPoses( const std::string& folder_with_poses,
                         std::map<std::string, Panorama>& pano_infos );