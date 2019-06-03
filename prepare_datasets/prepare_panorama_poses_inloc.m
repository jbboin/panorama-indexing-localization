clear all

[~,inloc_root] = system("cd ..;python -c 'import config;print(config.INLOC_ROOT)'");
inloc_root = deblank(inloc_root);

addpath('utils');

scans_path = fullfile(inloc_root, 'download/scans');
trans_path = fullfile(inloc_root, 'download/alignment');

buildings = {'DUC1','DUC2'};

for b = 1:numel(buildings)

    building = buildings{b};
    path_dataset = fullfile(scans_path, building);
    files = dir(fullfile(path_dataset, '*.mat'));
    
    incorrect_files = textread(fullfile(trans_path, building, 'know_incorrect.txt'), '%s');

    N_ref = numel(files);
    positions_ref = zeros(N_ref, 3);
    folder_pose = fullfile(inloc_root, building, 'pose');
    if ~exist(folder_pose)
        mkdir(folder_pose)
    end

    %% 
    % iterate over files, get transformation matrix, write JSON file
    for cnt = 1:numel(files)
        file = files(cnt);
        
        % check that the file is correct
        if any(strcmp(incorrect_files, file.name(end-10:end-8)))
            continue
        end
        
        fprintf('Loading %s...\n', file.name);
        TSCR = load(fullfile(path_dataset, file.name), 'T', 'S', 'Ncol', 'Nrow');
        S = TSCR.S;

        [filepath, scan_name, ext] = fileparts(file.name);
        [~, scan_name_clean, ~] = fileparts(scan_name);
        scan_name_trans = regexprep(scan_name_clean, 'scan', 'trans');

        file_transformation = fullfile(trans_path, building, 'transformations', [scan_name_trans, '.txt']); 
        fprintf('Reading from %s\n', file_transformation);
        fileID = fopen(file_transformation, 'r');
        transformation = textscan(fileID, '%f %f %f %f', 'HeaderLines', 7);
        transformation = cell2mat(transformation);
        fclose(fileID);

        position = transformation(1:3,4);
        Ncol = TSCR.Ncol;
        Nrow = TSCR.Nrow;

        positions_ref(cnt,:) = position;

        json_filename = sprintf('%s/%s.json', folder_pose, scan_name_clean);

        % still not sure what S means. More info here 
        % https://w3.leica-geosystems.com/kb/?guid=5532D590-114C-43CD-A55F-FE79E5937CB2
        scanner_position_local = S(1,:);
        scanner_axis_local = S(2:4,1:3);
        room = 'test_room'; % dummy room label


        struct_json = create_json_structure(Ncol, Nrow, transformation, scanner_position_local, scanner_axis_local, room);
        write_to_json(struct_json, json_filename);
    end

end
