%HW4
%% 
% Problem 1. 
% GB Comments:
1a 100
1b 100
1c 100
1d 100
2a 100
2b. 75 Code looks like it should work fine, but I am having difficulties getting the correct output. If you like, you can demonstrate on your computer that it works for full points. 
3a. 100
3b 100
3c 100
3d 100
3e 100
4a. 100
4b. 100 
 Overall = 98

% 1. Write a function to generate an 8-bit image of size 1024x1024 with a random value 
% of the intensity in each pixel. Call your image rand8bit.tif. 

write_random('rand8bit.tif')

% 2. Write a function that takes an integer value as input and outputs a
% 1024x1024 binary image mask containing 20 circles of that size in random
% locations

radius = 5;
circle_mask = cirimg(radius);

% 3. Write a function that takes the image from (1) and the binary mask
% from (2) and returns a vector of mean intensities of each circle (hint: use regionprops).

img = imread('rand8bit.tif');
circle_mi = cirintensity(img, circle_mask);

% 4. Plot the mean and standard deviation of the values in your output
% vector as a function of circle size. Explain your results. 

% Give a vector with radii to test, and initialize vectors for mean and
% standard deviations of mean intensities
radii = 5:5:100;
means = zeros(size(radii));
stds = zeros(size(radii));

counter = 0;
for radius = radii
    counter = counter + 1;
    
    % Make the circle mask and get mean intensities
    mask = cirimg(radius);
    circle_mi = cirintensity(img, mask);
    
    % Get mean and stddev of mean intensities and save
    means(counter) = mean(circle_mi);
    stds(counter) = std(circle_mi);
    
end

plot(radii, means, 'o')
axis([radii(1) radii(end) 100 150])
title('Mean of Mean Intensities of Circle Regions vs Radius')
xlabel('Radius')
ylabel('Mean of Mean Intensities')
figure
plot(radii, stds, 'o')
title('Standard Deviation of Mean Intensities of Circle Regions vs Radius')
xlabel('Radius')
ylabel('Stddev of Mean Intensities')

% The means of the mean intensities are very consistently arount 127. This
% is because the random image has an even distribution of intensity over
% its entire domain, as it is essenially just static. 

% The standard deviations of the mean intensities begins fairly large, and
% then becomes very small as the size of the circle becomes large. This is
% because of the law of large numbers. As more pixels are encompassed in
% the circle domain, it is more likely that the mean intensity will be
% equal to (or very close to) the average intensity of the entire static
% image. Small circles may find random irregularities in the intensities of
% the static, but larger circles should cover enough pixels so as to get a
% representative sample of the average pixel intensity of the random image.

%%

%Problem 2. Here is some data showing an NFKB reporter in ovarian cancer
%cells.
%https://www.dropbox.com/sh/2dnyzq8800npke8/AABoG3TI6v7yTcL_bOnKTzyja?dl=0
%There are two files, each of which have multiple timepoints, z
%slices and channels. One channel marks the cell nuclei and the other
%contains the reporter which moves into the nucleus when the pathway is
%active.
%
%Part 1. Use Fiji to import both data files, take maximum intensity
%projections in the z direction, concatentate the files, display both
%channels together with appropriate look up tables, and save the result as
%a movie in .avi format. Put comments in this file explaining the commands
%you used and save your .avi file in your repository (low quality ok for
%space).

% 1st, I imported both .tif files with Fiji using the command
% 'Plugins/Bio-Formats/Bio-Formats Importer'

% 2nd, I took maximum intensity projections in the z-direction with the
% Fiji command 'Image/Stacks/Z Project...' for each image, choosing the
% option 'Max Intensity'

% 3rd, I concatenated the files using the Fiji command
% 'Image/Stacks/Tools/Concatenate...', selecting each max intensity
% projection in the drop-down menu.

% 4th, I normalized each channel so that the image was visible using the
% Fiji command 'Process/Encance Contrast...' on each of the two channels.

% 5th, I displayed both channels together with an RGB lookup table with the
% Fiji command 'Image/Color/Make Composite' (channel 1 red, channel 2
% green)

% 6th, to save the video file generated, I used the Fiji command
% 'File/Save As/AVI...'


%Part 2. Perform the same operations as in part 1 but use MATLAB code. You don't
%need to save the result in your repository, just the code that produces
%it.

% Read in the images
reader1 = bfGetReader('nfkb_movie1.tif');
reader2 = bfGetReader('nfkb_movie2.tif');

% Take maximum intensity projections over z
nz = reader1.getSizeZ;

for t = 1:18
    
    ind1 = reader1.getIndex(0,0,t-1)+1;
    ind2 = reader1.getIndex(0,1,t-1)+1;
    ind3 = reader2.getIndex(0,0,t-1)+1;
    ind4 = reader2.getIndex(0,1,t-1)+1;
    img_max1 = bfGetPlane(reader1, ind1);
    img_max2 = bfGetPlane(reader1, ind2);
    img_max3 = bfGetPlane(reader2, ind3);
    img_max4 = bfGetPlane(reader2, ind4);
    
    for zz = 2:nz
        
        ind1 = reader1.getIndex(zz-1,0,t-1)+1;
        ind2 = reader1.getIndex(zz-1,1,t-1)+1;
        ind3 = reader2.getIndex(zz-1,0,t-1)+1;
        ind4 = reader2.getIndex(zz-1,1,t-1)+1;
        img_now1 = bfGetPlane(reader1, ind1);
        img_now2 = bfGetPlane(reader1, ind2);
        img_now3 = bfGetPlane(reader2, ind3);
        img_now4 = bfGetPlane(reader2, ind4);
        img_max1 = max(img_max1, img_now1);
        img_max2 = max(img_max2, img_now2);
        img_max3 = max(img_max3, img_now3);
        img_max4 = max(img_max4, img_now4);
        
    end
    
    timeslice1 = cat(3, imadjust(img_max1), imadjust(img_max2), zeros(size(img_max1)));
    timeslice2 = cat(3, imadjust(img_max3), imadjust(img_max4), zeros(size(img_max3)));
    
    if t == 1
        
        timepic1 = timeslice1;
        timepic2 = timeslice2;
        
    else
        
        timepic1 = cat(4, timepic1, timeslice1);
        timepic2 = cat(4, timepic2, timeslice2);
        
    end
    
end

% Take care of case t = 19 for first file
ind1 = reader1.getIndex(0,0,18)+1;
ind2 = reader1.getIndex(0,1,18)+1;
img_max1 = bfGetPlane(reader1, ind1);
img_max2 = bfGetPlane(reader1, ind2);

for zz = 2:nz
    
    ind1 = reader1.getIndex(zz-1,0,18)+1;
    ind2 = reader1.getIndex(zz-1,1,18)+1;
    img_now1 = bfGetPlane(reader1, ind1);
    img_now2 = bfGetPlane(reader1, ind2);
    img_max1 = max(img_max1, img_now1);
    img_max2 = max(img_max2, img_now2);
    
end

timeslice = cat(3, imadjust(img_max1), imadjust(img_max2), zeros(size(img_max1)));
timepic1 = cat(4, timepic1, timeslice);

% Concatenate the two image stacks into one image stack. This is the final
% video
timepic = cat(4, timepic1, timepic2);

%%

% Problem 3. 
% Continue with the data from part 2
% 
% 1. Use your MATLAB code from Problem 2, Part 2  to generate a maximum
% intensity projection image of the first channel of the first time point
% of movie 1. 

ind = reader1.getIndex(0,0,0)+1;
img_max = bfGetPlane(reader1, ind);

for zz = 2:nz
    
    ind = reader1.getIndex(zz-1,0,0)+1;
    img_now = bfGetPlane(reader1, ind);
    img_max = max(img_max, img_now);
    
end

% 2. Write a function which performs smoothing and background subtraction
% on an image and apply it to the image from (1). Any necessary parameters
% (e.g. smoothing radius) should be inputs to the function. Choose them
% appropriately when calling the function.

rad = 75;
sigma = 2;
img_signal = sm_bgsub(rad, sigma, img_max);

% 3. Write  a function which automatically determines a threshold  and
% thresholds an image to make a binary mask. Apply this to your output
% image from 2. 

mask = sig2mask(img_signal);

% 4. Write a function that "cleans up" this binary mask - i.e. no small
% dots, or holes in nuclei. It should line up as closely as possible with
% what you perceive to be the nuclei in your image. 

clean_mask = maskclean(mask);

% 5. Write a function that uses your image from (2) and your mask from 
% (4) to get a. the number of cells in the image. b. the mean area of the
% cells, and c. the mean intensity of the cells in channel 1. 

[num_cells, mean_area, mean_intensity] = cell_data(clean_mask, img_signal);

disp('Numbers of Cells:')
disp(num_cells)
disp('Mean Area:')
disp(mean_area)
disp('Mean Intensity:')
disp(mean_intensity)

% 6. Apply your function from (2) to make a smoothed, background subtracted
% image from channel 2 that corresponds to the image we have been using
% from channel 1 (that is the max intensity projection from the same time point). Apply your
% function from 5 to get the mean intensity of the cells in this channel.

ind = reader1.getIndex(0,1,0)+1;
img_max2 = bfGetPlane(reader1, ind);

for zz = 2:nz
    
    ind = reader1.getIndex(zz-1,1,0)+1;
    img_now = bfGetPlane(reader1, ind);
    img_max2 = max(img_max2, img_now);
    
end

% Use functions written for above parts to repeat experiment for channel 2
img_signal2 = sm_bgsub(rad, sigma, img_max2);
mask2 = sig2mask(img_signal2);
clean_mask2 = maskclean(mask2);
[~, ~, mean_intensity2] = cell_data(clean_mask2, img_signal2);

disp('Mean Intensity, Channel 2:')
disp(mean_intensity2)

%%
% Problem 4. 

% 1. Write a loop that calls your functions from Problem 3 to produce binary masks
% for every time point in the two movies. Save a movie of the binary masks.
% 

% Initialize matrices to hold mask image stacks and signal image stacks
mask_time = zeros(1024, 1024, 9);
mask_time2 = zeros(1024, 1024, 8);
mask_time3 = zeros(1024, 1024, 9);
mask_time4 = zeros(1024, 1024, 8);
signal_time = zeros(1024, 1024, 9);
signal_time2 = zeros(1024, 1024, 8);
signal_time3 = zeros(1024, 1024, 9);
signal_time4 = zeros(1024, 1024, 8);

rad = 75;
sigma = 2;
% Loop through time for each image stack
for t = 1:8
    
    % Get z max intensity projection
    ind = reader1.getIndex(0,0,t-1)+1;
    ind2 = reader2.getIndex(0,0,t-1)+1;
    ind3 = reader1.getIndex(0,1,t-1)+1;
    ind4 = reader2.getIndex(0,1,t-1)+1;
    img_max = bfGetPlane(reader1, ind);
    img_max2 = bfGetPlane(reader2, ind2);
    img_max3 = bfGetPlane(reader1, ind3);
    img_max4 = bfGetPlane(reader2, ind4);
    
    for zz = 2:nz
        
        ind = reader1.getIndex(zz-1,0,t-1)+1;
        ind2 = reader2.getIndex(zz-1,0,t-1)+1;
        ind3 = reader1.getIndex(zz-1,1,t-1)+1;
        ind4 = reader2.getIndex(zz-1,1,t-1)+1;
        img_now = bfGetPlane(reader1, ind);
        img_now2 = bfGetPlane(reader2, ind2);
        img_now3 = bfGetPlane(reader1, ind3);
        img_now4 = bfGetPlane(reader2, ind4);
        img_max = max(img_max, img_now);
        img_max2 = max(img_max2, img_now2);
        img_max3 = max(img_max3, img_now3);
        img_max4 = max(img_max4, img_now4);
        
    end
    
    % Get smoothed and background-subtracted image, and turn to clean mask
    img_signal = sm_bgsub(rad, sigma, img_max);
    img_signal2 = sm_bgsub(rad, sigma, img_max2);
    img_signal3 = sm_bgsub(rad, sigma, img_max3);
    img_signal4 = sm_bgsub(rad, sigma, img_max4);
    mask = sig2mask(img_signal);
    mask2 = sig2mask(img_signal2);
    mask3 = sig2mask(img_signal3);
    mask4 = sig2mask(img_signal4);
    clean_mask = maskclean(mask);
    clean_mask2 = maskclean(mask2);
    clean_mask3 = maskclean(mask3);
    clean_mask4 = maskclean(mask4);
    mask_time(:,:,t) = clean_mask;
    mask_time2(:,:,t) = clean_mask2;
    mask_time3(:,:,t) = clean_mask3;
    mask_time4(:,:,t) = clean_mask4;
    
end

% Repeat for t = 9 for first stack

% Get z max intensity projection
ind = reader1.getIndex(0,0,8)+1;
img_max = bfGetPlane(reader1, ind);
ind3 = reader1.getIndex(0,1,8)+1;
img_max3 = bfGetPlane(reader1, ind3);

for zz = 2:nz
    
    ind = reader1.getIndex(zz-1,0,8)+1;
    ind3 = reader1.getIndex(zz-1,1,8)+1;
    img_now = bfGetPlane(reader1, ind);
    img_now3 = bfGetPlane(reader1, ind3);
    img_max = max(img_max, img_now);
    img_max3 = max(img_max3, img_now3);
    
end

% Get smoothed and background-subtracted image, and turn to clean mask
img_signal = sm_bgsub(rad, sigma, img_max);
img_signal3 = sm_bgsub(rad, sigma, img_max3);
mask = sig2mask(img_signal);
mask3 = sig2mask(img_signal3);
clean_mask = maskclean(mask);
clean_mask3 = maskclean(mask3);

mask_time(:,:,9) = clean_mask;
mask_time3(:,:,9) = clean_mask3;

final_mask_video = cat(3, mask_time, mask_time2);
final_mask_video2 = cat(3, mask_time3, mask_time4);

% Write videos
v = VideoWriter('maskvideo_channel1');
open(v)
for t = 1:17
    writeVideo(v, final_mask_video(:,:,t));
end
close(v)

v = VideoWriter('maskvideo_channel2');
open(v)
for t = 1:17
    writeVideo(v, final_mask_video2(:,:,t));
end
close(v)

% 2. Use a loop to call your function from problem 3, part 5 on each one of
% these masks and the corresponding images and 
% get the number of cells and the mean intensities in both
% channels as a function of time. Make plots of these with time on the
% x-axis and either number of cells or intensity on the y-axis. 

% Initialize vectors to hold number of cells and mean intensities
num_cells_chan1 = zeros(1, 17);
num_cells_chan2 = zeros(1, 17);
mean_int_chan1 = zeros(1, 17);
mean_int_chan2 = zeros(1, 17);

% Loop through time and use stored masks, signal images
for t = 1:17
    
    if t < 10
        % Get max z intensity projections
        ind = reader1.getIndex(0,0,t-1)+1;
        img_max = bfGetPlane(reader1, ind);
        ind2 = reader1.getIndex(0,1,t-1)+1;
        img_max3 = bfGetPlane(reader1, ind2);
        
        for zz = 2:nz
            
            ind = reader1.getIndex(zz-1,0,t-1)+1;
            ind2 = reader1.getIndex(zz-1,1,t-1)+1;
            img_now = bfGetPlane(reader1, ind);
            img_now2 = bfGetPlane(reader1, ind2);
            img_max = max(img_max, img_now);
            img_max2 = max(img_max3, img_now2);
            
        end
        
    else
        % Get max z intensity projections
        ind = reader2.getIndex(0,0,t-10)+1;
        img_max = bfGetPlane(reader2, ind);
        ind2 = reader2.getIndex(0,1,t-10)+1;
        img_max3 = bfGetPlane(reader2, ind2);
        
        for zz = 2:nz
            
            ind = reader2.getIndex(zz-1,0,t-10)+1;
            ind2 = reader2.getIndex(zz-1,1,t-10)+1;
            img_now = bfGetPlane(reader2, ind);
            img_now2 = bfGetPlane(reader2, ind2);
            img_max = max(img_max, img_now);
            img_max2 = max(img_max3, img_now2);
            
        end
        
        
    end
    % Use cell data function to get number of cells, mean intensities
    [num_cells, ~, mean_intensity] = cell_data(logical(final_mask_video(:,:,t)), img_max);
    [num_cells2, ~, mean_intensity2] = cell_data(logical(final_mask_video2(:,:,t)), img_max);
    
    % Store data
    num_cells_chan1(t) = num_cells;
    num_cells_chan2(t) = num_cells2;
    mean_int_chan1(t) = mean_intensity;
    mean_int_chan2(t) = mean_intensity2;
    
end

% Make plots that were asked for
figure
plot(1:17, mean_int_chan1, 'o')
title('Mean Intensities of Channel 1 Over Time')
ylabel('Intensity Value')
xlabel('Time')
figure
plot(1:17, mean_int_chan2, 'o')
title('Mean Intensities of Channel 2 Over Time')
ylabel('Intensity Value')
xlabel('Time')


% FUNCTIONS

% Problem 1

function write_random( filename )

% Make an int8 image, size 1024, and save
img = im2double(rand(1024));
imwrite(img, filename)

end

function[ circle_mask ] = cirimg( radius )

% Define the size of the circle, and get locations
circle_img = im2double(zeros(1024));
xloc = randi(1024, 20, 1);
yloc = randi(1024, 20, 1);

% Iterate through each circle location and place a white dot in that
% location
for index = 1:20
    
    circle_img(xloc(index), yloc(index)) = 1;
    
end

% Dilate all the white dots to the desired radius.
circle_mask = imdilate(circle_img, strel('disk', radius));

end

function[ circle_mi ] = cirintensity( image, circle_domains )

% Use the circles to define domains for measuring mean intensities of the
% given image
circle_props = regionprops(logical(circle_domains), image, 'MeanIntensity');
circle_mi = [circle_props.MeanIntensity];

end

% Problem 3

function [img_signal] = sm_bgsub( rad, sigma, img )

% Smooth with Gaussing filter
fgauss = fspecial('gaussian', rad, sigma);
img_smooth = imfilter(img, fgauss);

% Subtract background
img_noise = imopen(img_smooth, strel('disk', 100));
img_signal = imsubtract(img_smooth, img_noise);


end

function [mask] = sig2mask( img )

% Determine threshold for image
threshold = max(max(img)) * 0.06;

% Get mask using threshold
mask = img > threshold;

end

function [clean_mask] = maskclean( mask )

% Perform dilation, then erosion, to get rid of small holes
img_close = imclose(mask, strel('disk', 4));

% Perform erosion, then dilation, to get rid of small spots
clean_mask = imopen(img_close, strel('disk', 3));

end

function [ num_cells, mean_area, mean_intensity ] = cell_data(mask, img)

% Use regionprops to get data on cells
cell_properties = regionprops(mask, img, 'Area', 'MeanIntensity');

% Get a. number of cells
num_cells = size([cell_properties.Area], 2);

% Get b. mean area of cells
mean_area = mean([cell_properties.Area]);

% Get c. mean intensity of each cell
mean_intensity = mean([cell_properties.MeanIntensity]);

end
