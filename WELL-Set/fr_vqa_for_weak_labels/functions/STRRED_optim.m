% The following papers are to be cited in the bibliography whenever the software is used
% as:
% C. G. Bampis, P. Gupta, R. Soundararajan and A. C. Bovik, "SpEED-QA: Spatial Efficient 
% Entropic Differencing for Image and Video Quality", Signal Processing
% Letters, under review
% R. Soundararajan and A. C. Bovik, "RRED indices: Reduced reference entropic
% differences for image quality assessment", IEEE Transactions on Image
% Processing, vol. 21, no. 2, pp. 517-526, Feb. 2012

function [sred_val, sred_val_sn, ...
    tred_val, tred_val_sn] = STRRED_optim(ref, ref_next, dis, ...
    dis_next, band, Nscales, Nor, ...
    blk, sigma_nsq)
%STRRED_OPT Summary of this function goes here
%   Detailed explanation goes here

helper_vct = Nor : -1 : 1;
ht = ceil((band - 1)/Nor);
which_or = band - ((ht - 1) * Nor + 1);
which_or = helper_vct(which_or);

%Wavelet decompositions using steerable pyramids
[pyr, pind] = buildSpyr_single(ref, ht, Nscales - 1, which_or, ...
    'sp5Filters', ...
    'reflect1');
dframe = ind2wtree(pyr, pind);
clear pyr
y1 = dframe{1, 2};

[pyr, pind] = buildSpyr_single(dis, ht, Nscales - 1, which_or, ...
    'sp5Filters', ...
    'reflect1');
dframe = ind2wtree(pyr, pind);
clear pyr
y2 = dframe{1, 2};

%estimate the entropy at different locations and the local spatial
%premultipliers
[ss_ref, q_ref] = est_params(y1, blk, sigma_nsq);%Spatial
spatial_ref = q_ref.*log2(1 + ss_ref);

[ss_dis, q_dis] = est_params(y2, blk, sigma_nsq);%Spatial
spatial_dis = q_dis.*log2(1 + ss_dis);

sred_val = nanmean(abs(spatial_ref(:)-spatial_dis(:)));
sred_val_sn = abs(nanmean(spatial_ref(:)-spatial_dis(:)));

ref_diff = ref - ref_next;
dis_diff = dis - dis_next;

[pyr, pind] = buildSpyr_single(ref_diff, ht, Nscales - 1, which_or, ...
    'sp5Filters', ...
    'reflect1');
dframe = ind2wtree(pyr, pind);
clear pyr
y_ref_diff = dframe{1, 2};

[pyr, pind] = buildSpyr_single(dis_diff, ht, Nscales - 1, which_or, ...
    'sp5Filters', ...
    'reflect1');
dframe = ind2wtree(pyr, pind);
clear pyr
y_dis_diff = dframe{1, 2};

%estimate the entropy at different locations and the local spatial
%premultipliers
[ss_ref_diff, q_ref] = est_params(y_ref_diff, blk, sigma_nsq);%Spatial
temporal_ref = q_ref.*log2(1 + ss_ref).*log2(1 + ss_ref_diff);

[ss_dis_diff, q_dis] = est_params(y_dis_diff, blk, sigma_nsq);%Spatial
temporal_dis = q_dis.*log2(1 + ss_dis).*log2(1 + ss_dis_diff);

tred_val = nanmean(abs(temporal_ref(:)-temporal_dis(:)));
tred_val_sn = abs(nanmean(temporal_ref(:)-temporal_dis(:)));

end

