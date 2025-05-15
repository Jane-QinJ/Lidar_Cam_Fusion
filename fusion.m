openExample('lidar/DetectionInLidarUsingCameraExample')
[imageDataLocation, lidarDataLocation] = helperDownloadWPIData();

% Load downloaded images data into the workspace
imds = imageDatastore(imageDataLocation);
imageFileNames = imds.Files;

% Load downloaded lidar data into the workspace
load(lidarDataLocation);

% Load calibration data
if ~exist('calib','var')
    load('calib.mat')
end

% Define camera to lidar transformation matrix
camToLidar = calib.extrinsics;
intrinsics = calib.intrinsics;
% Read an image frame into the workspace, and display it with the bounding boxes overlaid.
load imageGTruth.mat
im = imread(imageFileNames{50});
imBbox = imageGTruth{50};
figure
imshow(im)
showShape('rectangle',imBbox)
% insertShape(im, 'Rectangle', imBbox, 'Color', 'red', 'LineWidth', 3);

pc = lidarData{50};

% Crop point cloud to process only front region
roi = [0 70 -15 15 -3 8];
ind = findPointsInROI(pc,roi);
pc = select(pc,ind);

lidarBbox = bboxCameraToLidar(imBbox,pc,intrinsics, ...
    camToLidar,'ClusterThreshold',2,'MaxDetectionRange',[1,70]);
figure
pcshow(pc.Location,pc.Location(:,3))
showShape('Cuboid',lidarBbox)
view([-2.90 71.59])

% Initialize display
display = helperLidarCameraObjectsDisplay;
initializeDisplay(display)

% Update display with point cloud and image
updateDisplay(display, im, pc)

tic; % Start total timing
totalFrames = 200;

parpool; % Start parallel pool
parfor i = 1:200
    % Load point cloud and image
    im = imread(imageFileNames{i});
    pc = lidarData{i};

    % Load image ground truth
    imBbox = imageGTruth{i};

    % Remove ground plane
    groundPtsIndex = segmentGroundFromLidarData(pc,'ElevationAngleDelta',15, ...
        'InitialElevationAngle',10);
    nonGroundPts = select(pc,~groundPtsIndex);

    if imBbox
        [lidarBbox,~,boxUsed] = bboxCameraToLidar(imBbox,nonGroundPts,intrinsics, ...
            camToLidar,'ClusterThreshold',2,'MaxDetectionRange',[1, 50]);
        % Display image with bounding boxes
        im = updateImage(display,im,imBbox);
    end
    % Display point cloud with bounding box
    updateDisplay(display,im,pc);
    updateLidarBbox(display,lidarBbox,boxUsed)
    drawnow

    end

totalTime = toc; % End total timing
fps = totalFrames / totalTime;

fprintf('FPS: %.2f\n', fps);
% Initialize display
display = helperLidarCameraObjectsDisplay;
initializeDisplay(display)

% Get lidar to camera matrix
lidarToCam = invert(camToLidar);

% Loop first 200 frames. To loop all frames, replace 200 with numel(imageGTruth)
for i = 1:200
    im = imread(imageFileNames{i});
    pc = lidarData{i};
    imBbox = imageGTruth{i};

    % Remove ground plane
    groundPtsIndex = segmentGroundFromLidarData(pc,'ElevationAngleDelta',15, ...
        'InitialElevationAngle',10);
    nonGroundPts = select(pc,~groundPtsIndex);

    if imBbox
        [lidarBbox,~,boxUsed] = bboxCameraToLidar(imBbox,nonGroundPts,intrinsics, ...
            camToLidar,'ClusterThreshold',2,'MaxDetectionRange',[1, 70]);
        [distance,nearestRect,idx] = helperComputeDistance(imBbox,nonGroundPts,lidarBbox, ...
            intrinsics,lidarToCam);

        % Update image with bounding boxes
        im = updateImage(display,im,nearestRect,distance);
        updateLidarBbox(display,lidarBbox)
    end

    % Update display
    updateDisplay(display,im,pc)
    drawnow
end



function [distance, nearestRect, index] = helperComputeDistance(imBbox, pc, lidarBbox, intrinsic, lidarToCam)
% helperComputeDistance estimates the distance of 2-D bounding box in a given
% image using 3-D bounding boxes from lidar. It also calculates
% association between 2-D and 3-D bounding boxes

%   Copyright 2020 MathWorks, Inc.

numLidarDetections = size(lidarBbox,1);

nearestRect = zeros(0,4);
distance = zeros(1,numLidarDetections);
index = zeros(0,1);

for i = 1:numLidarDetections
    bboxCuboid = lidarBbox(i,:);

    % Create cuboidModel
    model = cuboidModel(bboxCuboid);

    % Find points inside cuboid
    ind = findPointsInModel(model,pc);
    pts = select(pc,ind);

    % Project cuboid points to image
    imPts = projectLidarPointsOnImage(pts,intrinsic,lidarToCam);

    % Find 2-D rectangle corresponding to 3-D bounding box
    [nearestRect(i,:),idx] = findNearestRectangle(imPts,imBbox);
    index(end+1) = idx;
    % Find the distance of the 2-D rectangle
    distance(i) = min(pts.Location(:,1));
end
end

function [nearestRect,idx] = findNearestRectangle(imPts,imBbox)
numBbox = size(imBbox,1);
ratio = zeros(numBbox,1);

% Iterate over all the rectangles
for i = 1:numBbox
    bbox = imBbox(i,:);
    corners = getCornersFromBbox(bbox);

    % Find overlapping ratio of the projected points and the rectangle
    idx = (imPts(:,1) > corners(1,1)) & (imPts(:,1) < corners(2,1)) & ...
        (imPts(:,2) > corners(1,2)) & (imPts(:,2) < corners(3,1));
    ratio(i) = sum(idx);
end

% Get nearest rectangle
[~,idx] = max(ratio);
nearestRect = imBbox(idx,:);
end

function cornersCamera = getCornersFromBbox(bbox)
cornersCamera = zeros(4,2);
cornersCamera(1,1:2) = bbox(1:2);
cornersCamera(2,1:2) = bbox(1:2) + [bbox(3),0];
cornersCamera(3,1:2) = bbox(1:2) + bbox(3:4);
cornersCamera(4,1:2) = bbox(1:2) + [0,bbox(4)];
end