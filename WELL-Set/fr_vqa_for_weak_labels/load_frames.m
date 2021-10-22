% 1. load frames from video file.
% 2. downscale frames by scale 2.
function [frms, frms_scaled] = load_frames(file, width, height, noof)
frms = zeros(height, width, noof);
frms_scaled = zeros(height/2, width/2, noof);

ffid = fopen(file, 'r');
for j = 1:noof
    fseek(ffid, 1.5 * width * height * (j-1), 'bof');
    frm = double(transpose(fread(ffid, [width, height], 'uint8')));

    frms(:,:,j) = frm;
    frms_scaled(:,:,j) = downscale(frm);    
end
fclose(ffid);

end

% downscale frames with scale by 2 using average pooling.
function v = downscale(frms)

[h, w, dims] = size(frms);

if dims == 1
    v = downscale_single(frms);
else
    v = zeros(h/2, w/2, dims);
    for i = 1:dims
        v(:,:,i) = downscale_single(frms(:,:,i));
    end
end
end

% downscale a single frame with scale by 2 using average pooling.
function frm = downscale_single(frm)
k = ones(2) / 4;
frm = conv2(frm, k, 'valid');
frm = frm(1:2:end, 1:2:end);
end