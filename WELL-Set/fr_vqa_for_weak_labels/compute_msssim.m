% compute the SSIM scores for given `ref` & `dst` features.
function score = compute_msssim(ref, dst)
dims = size(ref, 3);

score = zeros(dims, 1);
for i = 1:dims
    im1 = ref(:,:,i);
    im2 = dst(:,:,i);
    
    score(i) = func_msssim(im1, im2);
end

end


function score = func_msssim(img1, img2)

level = 5;
weight = [0.0448 0.2856 0.3001 0.2363 0.1333]';

downsample_filter = ones(2)./4;
im1 = double(img1);
im2 = double(img2);

mssim_array = zeros(level, 1);
mcs_array = zeros(level, 1);
for l = 1:level
    [mssim_array(l), mcs_array(l)] = func_ssim(im1, im2);
    
    filtered_im1 = imfilter(im1, downsample_filter, 'symmetric', 'same');
    filtered_im2 = imfilter(im2, downsample_filter, 'symmetric', 'same');
    
    im1 = filtered_im1(1:2:end, 1:2:end);
    im2 = filtered_im2(1:2:end, 1:2:end);
end

%   score = prod(mssim_array.^weight);
score = prod(mcs_array(1:level-1).^weight(1:level-1))*(mssim_array(level).^weight(level));
end


function [mssim, mcs] = func_ssim(img1, img2)
win = fspecial('gaussian', 11, 1.5);	%
K(1) = 0.01;										% default settings
K(2) = 0.03;										%

C1 = (K(1)*255)^2;
C2 = (K(2)*255)^2;
win = win/sum(sum(win));

mu1   = filter2(win, img1, 'valid');
mu2   = filter2(win, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(win, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(win, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(win, img1.*img2, 'valid') - mu1_mu2;

ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
cs_map = (2*sigma12 + C2)./(sigma1_sq + sigma2_sq + C2);

mssim = mean2(ssim_map);
mcs = mean2(cs_map);

end