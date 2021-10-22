% extract the ST-RRED features for a given video.
function strred = extract_strred(vid)

noof = size(vid, 3);

for j = 2:2:noof
    f1 = vid(:,:,j-1);
    f2 = vid(:,:,j);
    
    [sf, tf] = extract_strred_single(f1, f2);
    
    if j == 2
        [h0, w0] = size(sf);
        strred_sf = zeros(h0, w0, noof/2);
        strred_tf = zeros(h0, w0, noof/2);
    end
    strred_sf(:,:,j/2) = sf;
    strred_tf(:,:,j/2) = tf;
end

strred.s = strred_sf;
strred.t = strred_tf;
end

function [spatial_f, temporal_f] = extract_strred_single(prev, next)

band = 4;
Nscales = 5;
Nor = 6;
blk = 3;
sigma_nsq = 0.1;

[spatial_f, temporal_f] = STRRED_optim(prev, next, ...
    band, Nscales, Nor, blk, sigma_nsq);

end

% The following papers are to be cited in the bibliography whenever the software is used
% as:
% C. G. Bampis, P. Gupta, R. Soundararajan and A. C. Bovik, "SpEED-QA: Spatial Efficient
% Entropic Differencing for Image and Video Quality", Signal Processing
% Letters, under review
% R. Soundararajan and A. C. Bovik, "RRED indices: Reduced reference entropic
% differences for image quality assessment", IEEE Transactions on Image
% Processing, vol. 21, no. 2, pp. 517-526, Feb. 2012

function [spatial_f, temporal_f] = STRRED_optim(prev, next,...
    band, Nscales, Nor, blk, sigma_nsq)
%STRRED_OPT Summary of this function goes here
%   Detailed explanation goes here

helper_vct = Nor : -1 : 1;
ht = ceil((band - 1)/Nor);
which_or = band - ((ht - 1) * Nor + 1);
which_or = helper_vct(which_or);

%Wavelet decompositions using steerable pyramids
[pyr, pind] = buildSpyr_single(prev, ht, Nscales - 1, which_or, ...
    'sp5Filters', ...
    'reflect1');
dframe = ind2wtree(pyr, pind);
clear pyr
y1 = dframe{1, 2};

%estimate the entropy at different locations and the local spatial
%premultipliers
[ss_ref, q_ref] = est_params(y1, blk, sigma_nsq);%Spatial
spatial_f = q_ref.*log2(1 + ss_ref);



[pyr, pind] = buildSpyr_single(next, ht, Nscales - 1, which_or, ...
    'sp5Filters', ...
    'reflect1');
dframe = ind2wtree(pyr, pind);
clear pyr
y_ref_diff = y1 - dframe{1, 2};


%estimate the entropy at different locations and the local spatial
%premultipliers
[ss_ref_diff, q_ref] = est_params(y_ref_diff, blk, sigma_nsq);%Spatial
temporal_f = q_ref.*log2(1 + ss_ref).*log2(1 + ss_ref_diff);

end