% generate the pencil drawing
% Code: Hao Sun
% Date: 2022-04-18
% Generate the pencil drawing image based on the method from
%   "Combining Sketch and Tone for Pencil Drawing Production" Cewu Lu, Li Xu, Jiaya Jia 
%   International Symposium on Non-Photorealistic Animation and Rendering (NPAR 2012), June, 2012
% Reference
%   "Combining Sketch and Tone for Pencil Drawing Production" Cewu Lu, Li Xu, Jiaya Jia 
%   International Symposium on Non-Photorealistic Animation and Rendering (NPAR 2012), June, 2012
% the part of code from this paper
%% Define the variable
clc;clear all;
data_dir = './inputs';
texture_dir = './pencils';
out_dir = './results';
global isDebug
global img_list;
img_list = {};
%% initialization
% delete the previous output
del(out_dir)
% get the image
listing = dir(data_dir);
listing = listing(3:end,:);
numOfImage = size(listing,1);

%% Define the parameters
dirNum = 8;
ks = 1;
strokenDepth = 2;
backDepth = 1.1;%Trace the pencil background several times
renderDepth = 1.5;%Trace the generated texture rendering several times
omega = [25 40 18];

%% Define the pattern
% debug pattern show the image directly
isDebug = 0;

isDebug = 1;
start = 1;


%% main process
if(isDebug == 1)
    final = start;
else
    start = 1;
    final = numOfImage;
end

for i = start:final%numOfImage
    input_image = imread(fullfile(listing(i).folder,listing(i).name));
    for rand_num = 7:8  
        fprintf("%d_%d\n",i,rand_num);
        P = im2double(imread(sprintf('%s/pencil_%d.jpg',texture_dir,rand_num)));
        out_image = PencilDrawing(im, P,dirNum,ks,strokenDepth,backDepth,renderDepth,omega);
        if(isDebug == 1)
            img_list{1} = input_image;
            img_list{6} = out_image;
            figure,montage(img_list, 'Size', [1, 6],'Size',[2 3]);
        end
        imwrite(out_image,sprintf("%s/%02d_%02d.jpg",out_dir,i,rand_num),'jpg','Quality',95);
    end

end


function I = PencilDrawing(im, P,dirNum,ks,strokenDepth,backDepth,renderDepth,omega)
%   Generate the pencil drawing
%   "Combining Sketch and Tone for Pencil Drawing Production" Cewu Lu, Li Xu, Jiaya Jia 
%   International Symposium on Non-Photorealistic Animation and Rendering (NPAR 2012), June, 2012
%  
%   Paras:
%   @im           : the input image.
%   @P            : the pencil texture.
%   @dirNum       : the number of directions.
%   @ks           : the length of convolution line.
%   @strokenDepth : Trace the stroken several times
%   @backDepth    : Trace the pencil background several times
%   @renderDepth  : Trace the generated texture rendering several times
%   @omega        : Weights for tone map generation 
%

    %% Read the image
    [H, W, sc] = size(im);
    im = im2double(im);
    %% Convert from rgb to yuv when nessesary
    if (sc == 3)
        yuvIm = rgb2ycbcr(im);
        lumIm = yuvIm(:,:,1);
    else
        lumIm = im;
    end
    %% Generate the stroke map
    S = GenStroke(lumIm, ks, dirNum ,strokenDepth);
    global isDebug;
    global img_list;
    if(isDebug == 1)
        img_list{3} = S;
    end

    %% Generate the tone map
    J = GenToneMap(lumIm,omega); 
    global isDebug;
    global img_list;
    if(isDebug == 1)
        img_list{4} = J;
    end

    %% Process the pencil texture
    P = rgb2gray(P);
    P = P .^ backDepth;

    %% Generate the pencil map
    T = GenPencil(lumIm, P, J,renderDepth);
    global isDebug;
    global img_list;
    if(isDebug == 1)
        img_list{5} = T;
    end

    %% Compute the result
    lumIm = S .* T;

    if (sc == 3)
        yuvIm(:,:,1) = lumIm;
        I = ycbcr2rgb(yuvIm);
    else
        I = lumIm;
    end
end

function S = GenStroke(im, ks, dirNum,strokenDepth)
%   Compute the stroke map
%  
%   Paras:
%   @im        : input image ranging value from 0 to 1.
%   @ks        : kernel size.
%   @dirNum    : number of directions.
%   @gammaS    : the darkness of the stroke.
%
    
    %% Initialization
    [H, W, ~] = size(im);
    %% Smoothing
    im = medfilt2(im, [3 3]);
    
    %% Image gradient
    imX = [abs(im(:,1:(end-1)) - im(:,2:end)),zeros(H,1)]; % add the col in the right col
    imY = [abs(im(1:(end-1),:) - im(2:end,:));zeros(1,W)]; % add the row in the last row
%     imEdge = imX + imY; 
    imX = imX .^ 2;
    imY = imY .^ 2;
    % implement formula (1)
    imEdge = sqrt(imX + imY);
    imEdge = immultiply(imEdge,5);
    global isDebug;
    global img_list;
    if(isDebug == 1)
        %figure, imshow(S)
        img_list{2} = imEdge;
    end
    %% Convolution kernel with horizontal direction 
    kerRef = zeros(ks*2+1);
    kerRef(ks+1,:) = 1;

    %% Classification 
    % rotate the kernal by 180/dirNum degrees to calcuate the 8 -
    % direction convolution and calcuate the edge.
    response = zeros(H,W,dirNum);
    for n = 1 : dirNum
        % Eight direction convolution kernels are performed on the gradient map
        % implement formula (2)
        ker = imrotate(kerRef, (n-1)*180/dirNum, 'bilinear', 'crop');
        response(:,:,n) = conv2(imEdge, ker, 'same');
    end
    % index is a matrix by m*n, and each element is the max index by col.
    % the range is [1,dirNum], and the max value is this point direction.
    [~, index] = max(response,[], 3); 

    %% Create the stroke
    C = zeros(H, W, dirNum);
    for n = 1 : dirNum
        % index == n is a matrix and the element is 0 or 1.
        % implement formula (3)
        C(:,:,n) = imEdge .* (index == n);
    end
    
    Spn = zeros(H, W, dirNum);
    for n = 1 : dirNum
        % Convolution aggregates nearby pixels along direction
        ker = imrotate(kerRef, (n-1)*180/dirNum, 'bilinear', 'crop');
        Spn(:,:,n) = conv2(C(:,:,n), ker, 'same');
    end
    % Accumulate the elements of these dirNum matrices according to the third dimension
    Sp = sum(Spn, 3);
    Sp = (Sp - min(Sp(:))) / (max(Sp(:)) - min(Sp(:)));
    S = 1 - Sp;
    S = S .^ strokenDepth;
end

function J = GenToneMap(im,Omega)
%   Compute the tone map 'T'
%  
%   Paras:
%   @im        : input image ranging value from 0 to 1.
%   @Omega     : Weights for tone map generation.
%
    
    %% Parameters
    Ub = 225;
    Ua = 105;
    Mud = 90;
    DeltaB = 9;
    DeltaD = 11;

    %% Compute the target histgram
    histgramTarget = zeros(256, 1);
    total = 0;
    for ii = 0 : 255
        if ii < Ua || ii > Ub
            p = 0;
        else
            p = 1 / (Ub - Ua);
        end
        
        histgramTarget(ii+1, 1) = (...
            Omega(1) * 1/DeltaB * exp(-(255-ii)/DeltaB) + ...
            Omega(2) * p + ...
            Omega(3) * 1/sqrt(2 * pi * DeltaD) * exp(-(ii-Mud)^2/(2*DeltaD^2))) * 0.01;
        
        total = total + histgramTarget(ii+1, 1);
    end
    histgramTarget(:, 1) = histgramTarget(:, 1)/total;

    %% Smoothing
    im = medfilt2(im, [5 5]);
    
    %% Histgram matching
    J = histeq(im, histgramTarget);
    
    %% Smoothing
    G = fspecial('average', 10);
    J = imfilter(J, G,'same');
end

function T = GenPencil(im, P, J,renderDepth)
%   Compute the pencil map 'T'
%  
%   Paras:
%   @im          : input image ranging value from 0 to 1.
%   @P           : the pencil texture.
%   @J           : the tone map.
%   @renderDepth :Trace the generated texture rendering several times

    %% Parameters
    theta = 0.2;
    
    [H, W, ~] = size(im);

    %% Initialization
    P = imresize(P, [H, W]);
    P = reshape(P, H*W, 1);
    logP = log(P);
    logP = spdiags(logP, 0, H*W, H*W);
    
    J = imresize(J, [H, W]);
    J = reshape(J, H*W, 1);
    logJ = log(J);
    
    e = ones(H*W, 1);
    Dx = spdiags([-e, e], [0, H], H*W, H*W);
    Dy = spdiags([-e, e], [0, 1], H*W, H*W);
    
    %% Compute matrix A and b
    A = theta * (Dx * Dx' + Dy * Dy') + (logP)' * logP;
    b = (logP)' * logJ;
    
    %% Conjugate gradient
    beta = pcg(A, b, 1e-6, 60); 

    %% Compute the result
    beta = reshape(beta, H, W);
    beta = (beta - min(beta(:))) / (max(beta(:)) - min(beta(:)))*5;
    P = reshape(P, H, W);
    
    T = P .^ beta;
    T = T .^ renderDepth;
end

function del(dir_path)
listing = dir(dir_path);
listing = listing(3:end,:);
numOfImage = size(listing,1);
for i = 1:numOfImage%numOfImage
    file_path = fullfile(listing(i).folder,listing(i).name)
    delete(file_path)
end
end
