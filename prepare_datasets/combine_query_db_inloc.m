% clear all

[~,inloc_root] = system("cd ..;python -c 'import config;print(config.INLOC_ROOT)'");
inloc_root = deblank(inloc_root);

addpath('utils');

buildings = {'DUC1','DUC2'};

for b = 1:numel(buildings)

    building = buildings{b};
    disp(['Processing building ' building])

    db_view_poses = fullfile(inloc_root, building, 'views', 'view_poses.json');
    db_json_struct = jsondecode(fileread(db_view_poses));

    query_view_poses = fullfile(inloc_root, building, 'queries', 'view_poses.json');
    query_json_struct = jsondecode(fileread(query_view_poses));
    
    % Add ID offset to query images, to prevent colliding IDs
    id_offset = db_json_struct.images(end).id + 1;
    for i = 1:numel(query_json_struct.images)
        query_json_struct.images(i).id = query_json_struct.images(i).id + id_offset;
    end

    %% Write query_db_split.json

    json_str_split = struct;
    
    N_db = numel(db_json_struct.images);
    json_str_split.db_views = cell(N_db,1);
    for i = 1:N_db
        db = db_json_struct.images(i);
        json_str_split.db_views{i}.id = db.id;
        json_str_split.db_views{i}.filename = db.filename;
    end

    N_q = numel(query_json_struct.images);
    json_str_split.query_views = cell(N_q,1);
    for i = 1:N_q
        q = query_json_struct.images(i);
        json_str_split.query_views{i}.id = q.id;
        json_str_split.query_views{i}.filename = q.filename;
    end

    path_json = fullfile(inloc_root, building, 'query_db_split.json');
    write_to_json(json_str_split, path_json);

    %% Copy files in automated fashion

    out_folder = fullfile(inloc_root, building, 'view_total');
    if(exist(out_folder) ~= 7) 
        mkdir(out_folder);
    end

    % Copy DB files
    db_dir_total = fullfile(inloc_root, building, 'views', '*.png');
    db_files = dir(db_dir_total);
    for f_id = 1:numel(db_files)
        if(mod(f_id, 100)==0)
            fprintf('db %d/%d\n', f_id, numel(db_files));
        end
        file = db_files(f_id);
        f_out = fullfile(out_folder, file.name);
        if( exist(f_out, 'file')==2)
            continue;
        end
        f_in = fullfile(inloc_root, building, 'views', file.name);
        [SUCCESS,MESSAGE,MESSAGEID] = copyfile(f_in, f_out);
    end

    % Copy query files
    q_dir_total = fullfile(inloc_root, building, 'queries', '*.JPG');
    q_files = dir(q_dir_total);
    for f_id = 1:numel(q_files)
        if(mod(f_id, 100)==0)
            fprintf('queries %d/%d\n', f_id, numel(q_files));
        end
        file = q_files(f_id);
        f_out = fullfile(out_folder, file.name);
        if( exist(f_out, 'file')==2)
            continue;
        end
        f_in = fullfile(inloc_root, building, 'queries', file.name);
        [SUCCESS,MESSAGE,MESSAGEID] = copyfile(f_in, f_out);
    end

    %% Merge view poses of DB and Q

    out_json = fullfile(out_folder, 'view_poses.json');
    views_total_json_struct.images = [db_json_struct.images(:); query_json_struct.images(:)];
    views_total_json_struct.image_height = db_json_struct.image_height;
    views_total_json_struct.image_width = db_json_struct.image_width;
    views_total_json_struct.calibration_matrix = db_json_struct.calibration_matrix;
    write_to_json(views_total_json_struct, out_json);
    
end
