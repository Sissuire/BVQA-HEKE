% compute the ST-RRED scores for given `ref` & `dst` features.
function score = compute_strred(ref, dst)

srred_diff = abs(ref.s - dst.s);
trred_diff = abs(ref.t - dst.t);

dims = size(srred_diff, 3);
srred_diff = reshape(srred_diff, [], dims);
trred_diff = reshape(trred_diff, [], dims);
score = nanmean(srred_diff, 1) .* nanmean(trred_diff, 1);
end