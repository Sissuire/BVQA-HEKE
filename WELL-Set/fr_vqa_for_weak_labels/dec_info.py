import csv
from scipy.io import loadmat
import numpy as np
import pickle


def decode_info(fileid, file):
    disfile = file.split('/')[1]

    matinfo = loadmat('./log/{fileid}/{filename}.mat'.format(fileid=fileid, filename=disfile))
    s_msssim = matinfo['score_msssim'].reshape(-1).astype('float')
    s_gmsd = matinfo['score_gmsd'].reshape(-1).astype('float')
    s_stgmsd = matinfo['score_stgmsd'].reshape(-1).astype('float')
    s_strred = matinfo['score_strred'].reshape(-1).astype('float')

    s_vmaf = []
    csvfile = './log/{fileid}/{filename}.csv'.format(fileid=fileid, filename=disfile)
    with open(csvfile, 'r') as f:
        reader = csv.reader(f)

        for idx, line in enumerate(reader):
            if idx > 4 and idx < 255:
                s_vmaf.append(float(line[0].split("\"")[15]))
    s_vmaf = np.stack(s_vmaf).reshape(-1).astype('float')

    if len(s_msssim) == 250 and len(s_gmsd) == 250 and len(s_stgmsd) == 248 and len(s_strred) == 125 and len(s_vmaf) == 250:
        pass
    else:
        print(' ERROR Frames !!!!!! ====== ')
    return {'file': file,
            'fileid': fileid,
            'msssim': s_msssim,
            'gmsd': s_gmsd,
            'stgmsd': s_stgmsd,
            'strred': s_strred,
            'vmaf': s_vmaf}


if __name__ == '__main__':
    # dataset = loadmat('/mnt/disk/yongxu_liu/datasets/WELL_Set/run/data/dataset.info.mat')
    # info = dataset['info'][0]
    #
    # dec_info = []
    # for i in range(len(info)):
    #     i = 3432
    #     this = info[i][0][0]
    #     w = this[0][0][0]
    #     h = this[1][0][0]
    #
    #     thisfiles = this[2]
    #     files = []
    #     for j in range(len(thisfiles)):
    #         files.append(thisfiles[j].split()[0])
    #
    #     reffile = this[3][0]
    #     thisdist = this[4]
    #     dist = []
    #     for j in range(len(thisdist)):
    #         dist.append(thisdist[j].split()[0])
    #
    #     #
    #     fileid = reffile.split('/')[1][:-4]
    #
    #     print('{idx:04d}, {file}'.format(idx=i, file=fileid))
    #     for file in files:
    #         tmp = decode_info(fileid, file)
    #         dec_info.append(tmp)
    #
    # with open('fr_vqa.pkl', 'wb') as f:
    #     pickle.dump(dec_info, f)

    import pickle
    with open('/mnt/disk/yongxu_liu/datasets/WELL_Set/run/data/dataset.info.pkl', 'rb') as f:
        info = pickle.load(f)

    dec_info = []
    for i, key in enumerate(info.keys()):
        this = info[key]

        w = this['width']
        h = this['height']

        reffile = this['ref']
        fileid = reffile.split('/')[1][:-4]
        files = this['dis']

        #
        print('{idx:04d}, {file}'.format(idx=i, file=fileid))
        for file in files:
            tmp = decode_info(fileid, file)
            dec_info.append(tmp)

    with open('fr_vqa.pkl', 'wb') as f:
        pickle.dump(dec_info, f)