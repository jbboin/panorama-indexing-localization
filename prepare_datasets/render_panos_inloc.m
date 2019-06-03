clear all

[~,inloc_root] = system("cd ..;python -c 'import config;print(config.INLOC_ROOT)'");
inloc_root = deblank(inloc_root);

input_path = fullfile(inloc_root, 'download/scans');
align_path = fullfile(inloc_root, 'download/alignment');

buildings = {'DUC1','DUC2'};

for b = 1:numel(buildings)
    
    building = buildings{b};
    disp(['Processing building ' building])
    
    incorrect_files = textread(fullfile(align_path, building, 'know_incorrect.txt'), '%s');
    
    pano_dir = fullfile(inloc_root, building, 'panoramas');
    if ~exist(pano_dir)
        mkdir(pano_dir);
    end

    panoramas = dir(fullfile(input_path, building));

    for p = 3:numel(panoramas)
        panorama = panoramas(p).name;
        
        % check that the file is correct
        if any(strcmp(incorrect_files, panorama(end-10:end-8)))
            continue
        end
        
        disp(['  ' panorama])
        in_path = fullfile(input_path, building, panorama);
        out_path = fullfile(pano_dir, panorama);
        out_path = [out_path(1:end-8) '.png'];
        if exist(out_path)
            continue;
        end

        data = load(in_path);
        pixels = uint8([data.A{5} data.A{6} data.A{7}]);
        img = flipud(reshape(pixels, [data.Nrow, data.Ncol, 3]));
        imwrite(img, out_path);
    end

end
