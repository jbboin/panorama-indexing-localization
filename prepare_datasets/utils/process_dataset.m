function process_dataset(dim_pano, matterport_folder, dataset_name)
% this is the function we use to process datasets
%% Input parameters
pano_fold_out = 'panoramas';
fold_skybox = 'matterport_skybox_images';

%% read out pano to room labelling

rooms_labelling = 'house_segmentations/panorama_to_region.txt';
file_label_total = sprintf('%s/%s/%s', matterport_folder, dataset_name, rooms_labelling);
fid = fopen(file_label_total);
data = textscan(fid, '%d %s %s %s', 'delimiter', ' ');
fclose(fid);
pano_names = data{2};
rooms = data{4};
room_ids = data{3};

pano_name_to_room = containers.Map();
for i=1:length(pano_names)
    room_total = [rooms{i} '_' room_ids{i}];
    pano_name_to_room(pano_names{i}) = room_total;
end

%% get all the ids in the folder
ids = [];
Files=dir(sprintf('%s/%s/%s/*_skybox0_sami.jpg', matterport_folder, dataset_name, fold_skybox));
for k=1:length(Files)
   filename = Files(k).name;
   idstring = strsplit(filename, '_');
   id_ = idstring{1};
   ids{end+1} = id_;
   %fprintf('%s \n', filename);
end

%% wrapper to stitch
vx = [-pi/2 -pi/2 0 pi/2 pi -pi/2];
vy = [pi/2 0 0 0 0 -pi/2];
for id_ind=1:length(ids)
    fprintf('Generating pano %d/%d\n', id_ind, length(ids));
    id = ids{id_ind};
    fold_pano = sprintf('%s/%s/%s/', matterport_folder, dataset_name, pano_fold_out);
    
    if( isKey(pano_name_to_room, id)==false )
        fprintf('Dataset %s contains no panorama %s !!!\n', dataset_name, id);
        continue;
    end
    
    
    
    name_image_pano = sprintf('%s/camera_%s_%s.png', fold_pano, id, pano_name_to_room(id));
    
    
    if( exist(fold_pano)~=7)
        mkdir(fold_pano);
    end
    
    if( exist(name_image_pano)==2 ) 
        fprintf('File %s already exists\n', name_image_pano);
        continue;
    end
    
    
    %% stitch here
    sepImg = [];
    for a = 1:6
        filename_skybox = sprintf('%s/%s/%s/%s_skybox%d_sami.jpg', matterport_folder, dataset_name, fold_skybox, id, a-1);
        sepImg(a).img = im2double(imread(filename_skybox));
        sepImg(a).vx = vx(a);
        sepImg(a).vy = vy(a);
        sepImg(a).fov = pi/2 + 0.001;
        sepImg(a).sz = size(sepImg(a).img);
    end
    panoskybox = combineViews( sepImg, dim_pano(1), dim_pano(2) );

    %% write here
    
    imwrite(panoskybox, name_image_pano);
end
