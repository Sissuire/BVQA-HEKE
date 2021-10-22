This folder is to integrate the selected fr-vqa algorithms, which contains:
    1. MS-SSIM;
    2. GMSD;
    3. stGMSD;
    4. stRRED_opt.
-----
Others are:
    5. VMAF;
    6. FAST;

=========================

Procedure:

1. decode `ref`;
2. extract necessary information for `ref`;
3. decode `dst`;
4. extract corresponding information for `dst`;
5. compute the frame-by-frame quality values for `ref` & `dst` pair;
6. do 3)~5) recursively for the same `ref`;
7. do 1)~6) recursively for different `ref`;

