# panorama-indexing-localization

[**Efficient Panorama Database Indexing for Indoor Localization**](https://web.stanford.edu/~jbboin/doc/2019_panorama_indexing.pdf)

By Jean-Baptiste Boin (Stanford University), Dmytro Bobkov (Technical University of Munich), Eckehard Steinbach (Technical University of Munich), Bernd Girod (Stanford University)

*Paper under review*

## Dependencies

###  MATLAB
Refer to mathworks website.

### Python with installed dependencies  

**Prerequisites**:
- MATLAB

#### Anaconda

Note: we use Python2.7 together with [Anaconda](https://www.anaconda.com/distribution/#linux). 

Conda environment description for painless installation of all packages is given in conda_panorama.yml. Create a new environment by running the following command: `conda env create --file conda_panorama.yml` and thus all dependencies should be done. After that don't forget to run `conda activate panorama_test`.

#### Manual

In case you want to install it from source (not recommended!), you need the following projects:

- Python with installed dependencies (described in requirements.txt). 
- FAISS, install it by following instructions at https://github.com/facebookresearch/faiss/blob/master/INSTALL.md.
- Caffe with Python bindings. You can follow instructions from [here](https://gist.github.com/nikitametha/c54e1abecff7ab53896270509da80215), for example. 
Also make sure that the installation path is included in the environment variable `$PYTHONPATH`.

## Dataset data

### Matterport 3D dataset

Request access to the [Matterport3D dataset](https://github.com/niessner/Matterport) (instructions are given on that project page). This will grant you access to a download script: `download_mp.py`. You should download this file and copy it to the `prepare_datasets` directory in this current repository.

### InLoc dataset

We use [WUSTL Indoor RGBD dataset](https://cvpr17.wijmans.xyz/data) and [InLoc queries](http://www.ok.sc.e.titech.ac.jp/INLOC) (usually abbreviated as the InLoc dataset in the rest of this README) as well as the related data we will generate in this project. Refer to prepare_datasets/download_inloc.py file.

## Project setup

**Step 1**: Clone the repository.

    $ git clone https://github.com/jbboin/panorama-indexing-localization.git

**Step 2**: Copy the `config.example.py` file to your own `config.py` file that will contain your system-dependent paths that the project will use.

    $ cp config.example.py config.py

You will need to populate this new file with your own desired paths. (**NOTE:** The paths should be absolute paths.)
- `DEEP_RETRIEVAL_ROOT`: Directory that will contain the CNN models used to extract the image descriptors.
- `INLOC_ROOT`: Directory where we will store the [WUSTL Indoor RGBD dataset](https://cvpr17.wijmans.xyz/data) and [InLoc queries](http://www.ok.sc.e.titech.ac.jp/INLOC) (usually abbreviated as the InLoc dataset in the rest of this README) as well as the related data we will generate in this project.
- `MATTERPORT_ROOT`: Directory where we will store the Matterport3D dataset (distractor dataset) as well as the related data we will generate in this project.

**Step 3**: Download the models and weights for the CNN feature extractor. We use the models from the [Deep Image Retrieval](https://europe.naverlabs.com/Research/Computer-Vision/Learning-Visual-Representations/Deep-Image-Retrieval) project. The download link is [here](http://download.europe.naverlabs.com/Computer-Vision-CodeandModels). Download `deep_retrieval.tgz` and extract it at the location specified in `DEEP_RETRIEVAL_ROOT` in your `config.py` file.

**Step 4**: Download the `download_mp.py` Matterport download script and copy it to the `prepare_datasets` directory in this current repository.

**Step 5**: This repository includes a lightly modified version of the FLANN library, which allows us to get access to the internal structure of the built trees from our Python codes. We only modified the file `flann/src/cpp/flann/algorithms/kmeans_index.h`. Build this library as a CMake project. We expect that the build directory will be located at `flann/build`. The following commands can be used to build the project as we expect.

    $ cd flann
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    $ cd ../.. # return to the root directory after this

**Step 6**: Add the modified FLANN library to the environment variable `$PYTHONPATH`.

    $ export PYTHONPATH=$(pwd)/flann/src/python:$PYTHONPATH

**Step 7**: Build the `view_rendering` CMake project included in this repository. We expect that the build directory will be located at `view_rendering/build`. The following commands can be used to build the project as we expect.

    $ cd view_rendering
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    $ cd ../.. # return to the root directory after this

Note that the project has the following dependencies: Eigen3, Boost1.4 and OpenCV.

Tested on Ubuntu 16.04 64-bit and on Mac.

## Dataset preparation

The scripts for this part are located in `prepare_datasets` and they are expected to be run from the root of this repository.

**Step 1**: Download and extract both datasets (InLoc and Matterport) using the following scripts.

    $ python prepare_datasets/download_inloc.py
    $ ./download_matterport.sh

**Step 2**: *Using MATLAB*, run the following scripts that will generate the panorama images for both datasets. They will be stored in the new directories `$INLOC_ROOT/DUC1/panoramas` and `$INLOC_ROOT/DUC2/panoramas` for the InLoc dataset, and `$MATTERPORT_ROOT/$building/panoramas` for the Matterport dataset, where `$building` is the ID of either of the 5 buildings that we use from the Matterport dataset to create our own distractor dataset. Note: we use [PanoBasic toolbox](https://github.com/yindaz/PanoBasic)  to render panoramas for Matterport dataset.

    $ run('prepare_datasets/render_panos_inloc.m')
    $ run('prepare_datasets/render_panos_matterport.m')

**Step 3**: *Using MATLAB*, run the following script. It will create the directories `$INLOC_ROOT/DUC1/pose` and `$INLOC_ROOT/DUC2/pose` that will contain a `json` file for each panorama of the InLoc dataset where information for that panorama (camera pose) is saved in a convenient format.

    $ run('prepare_datasets/prepare_panorama_poses_inloc.m')

**Step 4**: *Using MATLAB*, run the following script. It will resize the query images of the InLoc dataset and save them to the new directories `$INLOC_ROOT/DUC1/queries` and `$INLOC_ROOT/DUC2/queries`, depending on which part of the dataset a query corresponds to. It will also generate a metadata file `view_poses.json` in each directory where the poses of the queries is recorded. The pose information comes from the reference file `data/DUC_refposes_all.mat`, which was provided to us by the authors of the InLoc dataset.

    $ run('prepare_datasets/prepare_queries_inloc.m')

**Step 5**: The following script uses the executable built in Step 5 of the previous part to sample and render a fixed number of limited field-of-view views for each panorama. This is done for both InLoc and Matterport datasets. At the end a new directory `views` should be created for each sub-dataset. This directory will contain the rendered views (we use the parameters from our paper, so 144 views are generated per panorama) as well as a metadata file `view_poses.json`.

    $ ./prepare_datasets/render_views.sh 

**Step 6**: *Using Matlab*, run the following script that will combine the query and database parts of each sub-dataset of the InLoc dataset (`DUC1` and `DUC2`) to a single directory `view_total` that contains all images and a combined metadata file `view_poses.json`. This script also generates a file `query_db_split.json` that lists which images are part of the query set and which ones make up the database.

    $ run('prepare_datasets/combine_query_db_inloc.m')

**Step 7**: We now need to add ground truth labels for the rooms assigned to each panorama and query in the InLoc dataset (room labels are already included in the Matterport dataset). This is done by running the following script that reads a file we provide, `data/room_labels.csv`, containing the labels obtained with our method. The metadata files `$INLOC_ROOT/DUC1/view_total/view_poses.json` and `$INLOC_ROOT/DUC2/view_total/view_poses.json` will be updated with these labels. The code we used to generate these labels is not included in this repository since this is not the focus of this work, but you can use your own methods to generate these labels, as long as you generate a new file `data/room_labels.csv` containing your results in the same format.

    $ python prepare_datasets/add_room_labels_inloc.py

**Step 8**: Run the following script to extract descriptors from all images. For each sub-dataset of both InLoc and Matterport datasets, this will produce a single numpy array saved in a new `features` directory.

    $ ./prepare_datasets/extract_features.sh

**Step 9**: The following script will combine all buildings of the Matterport dataset to generate the combined distractor dataset that will be saved at `$MATTERPORT_ROOT/distractor_dataset`. After running this script, this directory should contained the combined numpy file containing the image descriptors as well as a combined metadata file `view_poses.json`.

    $ python prepare_datasets/combine_datasets_matterport.py

**Step 10**: This last script is used to generate extra metadata files named `query_db_split_samp_xx_yy.json` that are used for our dataset sub-sampling experiments. They are similar to the files `query_db_split.json` generated in Step 6 but only contain a subset of the views for the dataset part. The numbers `xx` and `yy` correspond to the number of views sampled horizontally and vertically, respectively.

    $ python prepare_datasets/dataset_resampling.py


## Evaluation

The scripts for this part are located in `evaluation_scripts` and they are expected to be run from the root of this repository. They can be used to replicate the results from the experiments we reported in our work. Note that the results may not perfectly match ours due to differences in library versions or randomness (for example the FLANN indexes that are generated may be different from ours, etc.), but they should be close enough for all practical purposes. For each set of results we provide the corresponding script that should be run to generate them.

The actual values that are printed in the console usually consist of a set of 3 values: the number of descriptors, the mAP and the precision at 1. The mAP was the metric reported in our work, and these scripts will separately report results for the `DUC1` and `DUC2` part of the InLoc dataset. As mentioned, in order to get the final mAP results, it is necessary to compute a weighted average using the number of queries for each part. As such, the results that we reported are:

    N1 = 130 # num queries in DUC1
    N2 = 198 # num queries in DUC2
    mAP = (N2 * mAP(DUC1) + N1 * mAP(DUC2)) / (N1 + N2)

**Step 1**: Sub-sampling experiment. Results reported in Sec. III.B. and Fig. 4 of the paper.

    $ ./evaluation_scripts/dataset_subsampling_evaluation.sh

**Step 2**: Aggregation experiment. Results reported in Sec. III.B. and Fig. 4 of the paper.

    $ ./evaluation_scripts/dataset_aggregation_evaluation.sh

**Step 3**: Evaluation of the autotune algorithm of the FLANN library. These results are not explicitly mentioned in our work but provide empirical justification that k-means tree indexes are preferable to kd-tree indexes for our data.

    $ ./evaluation_scripts/flann_autotune.sh

**Step 3**: FLANN evaluation. Results reported in Sec. III.C.1 and Fig. 5 of the paper.

    $ ./evaluation_scripts/flann_evaluation.sh

**Step 5**: DBH evaluation. Results reported in Sec. III.C.1 and Fig. 5 of the paper.

    $ ./evaluation_scripts/dbh_evaluation.sh

**Step 6**: Number of rooms recommended for each part of the InLoc dataset (This is used for the SBH index and the values are hardcoded in the SBH evaluation script, see next step). Results reported in Sec. III.C.2 of the paper.

    $ python evaluation_scripts/get_number_rooms.py

**Step 7**: SBH evaluation. Results reported in Sec. III.C.2 and Fig. 6 of the paper.

    $ ./evaluation_scripts/sbh_evaluation.sh

**Step 8**: PQ evaluation. Results reported in Sec. III.D and Table II of the paper.

    $ ./evaluation_scripts/pq_evaluation.sh
