clear;
close all;
folder = 'd2';

savepath = 'train.h5';

size_label = 96;
scale = 4;
size_input = size_label/scale;
stride = 48;
%% downsizing
% downsizes = [1,0.7,0.5];

data = zeros(size_input, size_input, 1, 1);
label_x4 = zeros(size_label, size_label, 1, 1);

count = 0;
margain = 0;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];


length(filepaths)

tic
for i = 1 : length(filepaths)
    image = imread(fullfile(folder,filepaths(i).name));
    if size(image,3)==3
        image = rgb2ycbcr(image);
        image = im2double(image(:, :, 1));  %??ycbcr????y??
        im_label = modcrop(image, scale);  %???????
        [hei,wid] = size(im_label);
        for x = 1 + margain : stride : hei-size_label+1 - margain
            for y = 1 + margain :stride : wid-size_label+1 - margain
                subim_label = im_label(x : x+size_label-1, y : y+size_label-1);
                subim_input = imresize(subim_label,1/scale,'bicubic');

                count=count+1;
                data(:, :, :, count) = subim_input;          
                label_x4(:, :, :, count) = subim_label;
            end
        end
    end
end

order = randperm(count);
data = data(:, :, :, order);
label_x4 = label_x4(:, :, :, order);

disp({'Generate',int2str(count)})   %???count?????

%% writing to HDF5
chunksz = 256;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)    % 1-10?batch

    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    size(batchdata)
    batchlabs_x4 = label_x4(:,:,:,last_read+1:last_read+chunksz);
    size(batchlabs_x4)

    
    startloc = struct('dat',[1,1,1,totalct+1], 'lab_x4', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs_x4, ~created_flag, startloc, chunksz);
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);

toc