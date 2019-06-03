clear all

[~,matterport_root] = system("cd ..;python -c 'import config;print(config.MATTERPORT_ROOT)'");
matterport_root = deblank(matterport_root);

addpath('utils');

files = dir(matterport_root);

for cnt = 1:numel(files)
    d = files(cnt).name;
    if numel(d) <= 2
        continue
    end
    disp(['Render panos for dataset: ' d])
    process_dataset([4096, 2048], matterport_root, d);
end
