% save fr-vqa scores for every video file.
function save_info(dst, score_msssim, score_gmsd, score_stgmsd, score_strred)
file = [dst, '.mat'];
save(file, 'score_msssim', 'score_gmsd', 'score_stgmsd', 'score_strred', ...
    '-v7');
end