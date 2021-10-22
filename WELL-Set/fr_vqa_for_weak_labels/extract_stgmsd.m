% extract the ST-GMSD features for a given video.
function feat = extract_stgmsd(vid)
[dx, dy, dt] = funcGetKernel();

gx = convn(vid, dx, 'valid');
gy = convn(vid, dy, 'valid');
gt = convn(vid, dt, 'valid');
feat = sqrt(gx.^2+gy.^2+gt.^2);
end

function [dx, dy, dt] = funcGetKernel()
basis_1 = [1 1 1; 1 1 1; 1 1 1] / 9;
basis_2 = zeros(3);
basis_3 = -basis_1;

dx(:, 1, :) = basis_1;
dx(:, 2, :) = basis_2;
dx(:, 3, :) = basis_3;

dy(1, :, :) = basis_1;
dy(2, :, :) = basis_2;
dy(3, :, :) = basis_3;

dt = cat(3, basis_1, basis_2, basis_3);
end