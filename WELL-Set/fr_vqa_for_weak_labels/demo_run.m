clear
close all
clc

root_path = '/mnt/disk/yongxu_liu/datasets/WELL_Set_run/run/data/';
load([root_path, 'dataset.info.mat']);  % the info data for WELL-Set

% for st-RRED_opt
addpath(genpath('matlabPyrTools'))
addpath(genpath('functions'))
% model = 'vmaf_v0.6.1.pkl';

if ~exist('tmp_path', 'dir')
    mkdir('tmp_path');
end

for jr = 1:length(info)
    
    % orig
    cur = info{jr};
    
    % convert to YUV420p
    str = regexp(cur.ref, '/', 'split');
    ids = str{2}(1:end-4);
    log_path = ['log/', ids, '/'];
    if ~exist(log_path, 'dir')
        mkdir(log_path);
    end
    
    %
    ref_file = [root_path, cur.ref];
    
    width = cur.width;
    height = cur.height;
    noof = 250;
    
    %% TODO: ssim -> MS-SSIM
%     tic
    [ref, ref_scaled] = load_frames(ref_file, width, height, noof);  %% NO
    
    rf_strred = extract_strred(ref);
    rf_msssim = extract_msssim(ref);
    rf_gmsd   = extract_gmsd(ref_scaled);
    rf_stgmsd = extract_stgmsd(ref_scaled);
    clear ref ref_scaled
    
    % dist
    
    for idis = 1:size(cur.dis, 1)
        str = regexp(cur.dis(idis, :), ' ', 'split');
        dst_file = [root_path, str{1}];
        str = regexp(str{1}, '/', 'split');
        id_dis = str{end};
        
        fprintf('\n----------\nProcessing %04d/%03d  %s\n', jr, idis, id_dis);
        
        str = regexp(cur.dist_type(idis, :), ' ', 'split');
        distype = str{1};
        
        % convert to YUV420p
        tmp_file = ['tmp_path/', id_dis, '.yuv'];
        if strcmp(distype, 'MPEG4_abr') || strcmp(distype, 'MPEG4_qscale')
            encoder = '-vcodec mpeg4 ';
        elseif strcmp(distype, 'MPEG2')
            encoder = '-vcodec mpeg2video ';
        elseif strcmp(distype, 'MJPEG')
            encoder = '-vcodec mjpeg ';
        elseif strcmp(distype, 'MJ2K')
            encoder = '-vcodec jpeg2000 ';
        elseif strcmp(distype, 'Snow')
            encoder = '-vcodec snow ';
        else
            encoder = '';
        end
        fprintf('%s   ---  %s\n', distype, encoder);
        command = ['ffmpeg -y ', encoder, '-i ', dst_file, ' -pix_fmt yuv420p -f rawvideo ', tmp_file];
        system(command);
        
        [dst, dst_scaled] = load_frames(tmp_file, width, height, noof);
        
        df_strred = extract_strred(dst);
        df_msssim = extract_msssim(dst);
        df_gmsd   = extract_gmsd(dst_scaled);
        df_stgmsd = extract_stgmsd(dst_scaled);
        clear dst dst_scaled
        
        score_msssim = compute_msssim(rf_msssim, df_msssim);
        score_gmsd   = compute_gmsd(rf_gmsd, df_gmsd);
        score_stgmsd = comptue_stgmsd(rf_stgmsd, df_stgmsd);
        score_strred = compute_strred(rf_strred, df_strred);
        
%         toc
        
        save_prefix = [log_path, id_dis];
        save_info(save_prefix, ...
            score_msssim, score_gmsd, score_stgmsd, score_strred);
        
        % vmaf
%         cmd = ['~/build_vmaf/libvmaf/build/tools/vmafossexec ', ...
%             'yuv420p ', num2str(width), ' ', num2str(height), ' ', ...
%             ref_file, ' ', tmp_file, ...
%             ' ~/build_vmaf/libvmaf/model/', model,...
%             ' --log ', save_prefix, '.csv ', '--log_fmt csv'];
%         system(cmd);
        fprintf('\nAll Done\n');
        
        delete(tmp_file);
    end
end

