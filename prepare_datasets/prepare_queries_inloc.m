clear all

[~,inloc_root] = system("cd ..;python -c 'import config;print(config.INLOC_ROOT)'");
inloc_root = deblank(inloc_root);

addpath('utils');

query_path = fullfile(inloc_root, 'download/iphone7');

buildings = {'DUC1','DUC2'};

query_pose_data = load('../data/DUC_refposes_all.mat');
DUC_queries = {query_pose_data.DUC1_RefList, query_pose_data.DUC2_RefList};

width_new = 640;
height_new = 480;

for b = 1:numel(buildings)

    building = buildings{b};
    disp(['Processing building ' building])

    fold_out = fullfile(inloc_root, building, 'queries');

    % save query positions in global coordinate system
    queries_to_process = DUC_queries{b};
    N_queries = numel(queries_to_process);
    positions_queries = zeros(N_queries, 3);
    rotations_queries = zeros(N_queries, 4); % rotation matrices
    for i=1:N_queries
        query = queries_to_process(i);    
        position_proj = query.P(1:3,4);
        orientation_proj = query.P(1:3,1:3);

        % transform from projection to global position, orientation
        % see
        % https://math.stackexchange.com/questions/82602/how-to-find-camera-position-and-rotation-from-a-4x4-matrix
        % if you need more info on this
        position_global = -orientation_proj' * position_proj;
        rotation = orientation_proj';
        orientation = orientation_proj' * [0; 0; 1];
        quat = rotm2quat(rotation);

        % no further transformation needed because the position seem to be in
        % global coordinate system of DUC1 or 2!
        positions_queries(i,:) = position_global;
        rotations_queries(i,:) = quat;
    end

    if(exist(fold_out) ~= 7) 
        mkdir(fold_out);
    end

    % write to JSON poses
    json_struct.image_height = height_new;
    json_struct.image_width = width_new;
    json_struct.calibration_matrix = zeros(1,9);

    images = cell(N_queries,1);
    for i=1:N_queries
        query = queries_to_process(i);
        image.camera_location = positions_queries(i,:);
        image.filename = query.queryname;
        image.id = i-1;
        image.ind_yaw = -1;
        image.room = "test_room";
        image.source_pano_filename = query.queryname;
        image.final_camera_rotation = rotations_queries(i,:);
        images{i} = image;
    end

    json_struct.images = images;

    path_json = fullfile(fold_out, 'view_poses.json');
    write_to_json(json_struct, path_json);

    parfor i=1:N_queries
        fprintf('%d/%d\n', i, N_queries);
        query = queries_to_process(i);
        file_in = fullfile(query_path, query.queryname);
        file_out = fullfile(fold_out, query.queryname);
        im_in = imread(file_in);
        im_out = imresize(im_in, [height_new, width_new]);
        imwrite(im_out, file_out, 'Quality', 100);
    end

end
