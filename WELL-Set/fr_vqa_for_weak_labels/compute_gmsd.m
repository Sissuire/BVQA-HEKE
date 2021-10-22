% compute the GMSD scores for given `ref` & `dst` features.
function score = compute_gmsd(ref, dst)

dims = size(ref, 3);
score = zeros(dims, 1);

for i = 1:dims
    gm1 = ref(:,:,i);
    gm2 = dst(:,:,i);
    quality_map = (2*gm1.*gm2 + 255) ./(gm1.^2+gm2.^2 + 255);
    score(i) = std2(quality_map);
end
end