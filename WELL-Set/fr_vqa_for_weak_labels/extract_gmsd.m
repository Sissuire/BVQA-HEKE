% extract the GMSD features for a given video.
function feat = extract_gmsd(vid)
dx = [1 0 -1; 1 0 -1; 1 0 -1]/3;
dy = dx';

[h, w, dims] = size(vid);
feat = zeros(h-2, w-2, dims);

for i = 1:dims
    Y = vid(:,:,i);
    Ix = conv2(Y, dx, 'valid');
    Iy = conv2(Y, dy, 'valid');
    feat(:,:,i) = sqrt(Ix.^2 + Iy.^2);
end
end