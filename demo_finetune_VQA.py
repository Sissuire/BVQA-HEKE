# Code for intra-dataset performance 
# -----------------------------------------------
# contact via `yongxu.liu@stu.xidian.edu.cn`

import os
import random
import numpy as np
import argparse
import timeit
import pickle
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from scipy import stats
from torch.utils.data import Dataset, DataLoader
from itertools import combinations

from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class PreDataset(object):
    r"""
    A pre-processing for video dataset: split train-validation set

    args:
        db_path: path to database
        info_file: info file
        database: database to use
        target: pre-fixed validation ref index
        data_path: pre-prepared data for loading
    """
    def __init__(self, args):
        self.tr_te_r = 0.2
        self.target = args.target
        self.database = args.database

        self.data = self.read_info(args)

    def read_info(self, args):
        info_file = args.info_file
        base = '{}/{}/'.format(args.base_path, args.database)
        ## read regular info
        ref_list, dis_list, d2r_list, score_list, width_list, height_list, fps_list, frms_list = \
            [], [], [], [], [], [], [], []
        with open(info_file, 'r') as f:
            for line in f:
                scn_idx, dis_idx, ref, dis, score, width, height, frm = line.split()
                scn_idx = int(scn_idx)
                dis_idx = int(dis_idx)
                width = int(width)
                height = int(height)
                frames = int(frm)
                fps = 30 if self.database == 'LIVE-Mobile' else int(frm) // 10

                dis_list.append(base + dis)
                ref_list.append(base + ref)
                d2r_list.append(scn_idx)
                score_list.append(float(score))
                width_list.append(width)
                height_list.append(height)
                fps_list.append(fps)
                frms_list.append(frames)

        ref_list = np.asarray(ref_list)
        dis_list = np.asarray(dis_list)
        d2r_list = np.array(d2r_list, dtype='int')
        score_list = np.array(score_list, dtype='float32')
        width_list = np.array(width_list, dtype='int')
        height_list = np.array(height_list, dtype='int')
        fps_list = np.array(fps_list, dtype='int')
        frms_list = np.array(frms_list, dtype='int')

        # DMOS
        score_list = 1 + 8.5 * (score_list - score_list.min()) / (score_list.max() - score_list.min())

        with open(args.load, 'rb') as f:
            data = pickle.load(f)

            difference = np.mean(np.abs(score_list - (data['mos'])))
            if difference < 1e-10:
                print('Check data order successfully {}'.format(u'\u2713'))
            else:
                print('data order seems wrong. please check.')
                raise NotImplementedError

            data = self.prepare(args, data['features'])
        if args.mos:
            print('reverse MOS to DMOS')
            score_list = 10.5 - score_list

        ## split
        scenes = np.unique(d2r_list)
        n_scenes = len(scenes)
        n_test = round(n_scenes * self.tr_te_r)

        if self.target is None:
            eval_enum = random.sample(range(n_scenes), n_test)
            self.target = eval_enum
        else:
            eval_enum = self.target

        print('eval_idx: ', eval_enum)
        train_idx, eval_idx, train_ref, eval_ref = self.split_data(d2r_list, eval_enum)
        print('number of training: {}, number of testing: {}'.format(len(train_idx), len(eval_idx)))

        train_dict = {'ref': ref_list[train_idx],
                      'dis': dis_list[train_idx],
                      'data': data[train_idx],
                      'mos': score_list[train_idx],
                      'width': width_list[train_idx],
                      'height': height_list[train_idx],
                      'fps': fps_list[train_idx],
                      'frms': frms_list[train_idx]}
        eval_dict = {'ref': ref_list[eval_idx],
                     'dis': dis_list[eval_idx],
                     'data': data[eval_idx],
                     'mos': score_list[eval_idx],
                     'width': width_list[eval_idx],
                     'height': height_list[eval_idx],
                     'fps': fps_list[eval_idx],
                     'frms': frms_list[eval_idx]}

        return {'train': train_dict, 'test': eval_dict}

    def prepare(self, args, data):
        n_vid = len(data)
        n_snp = len(data[0][0])
        f1 = np.zeros((n_vid, n_snp, 64*3), dtype=np.float32)
        f2 = np.zeros((n_vid, n_snp, 128*3), dtype=np.float32)
        f3 = np.zeros((n_vid, n_snp, 256*3), dtype=np.float32)
        f4 = np.zeros((n_vid, n_snp, 512*3), dtype=np.float32)

        for i, d in enumerate(data):
            f1[i] = d[0]
            f2[i] = d[1]
            f3[i] = d[2]
            f4[i] = d[3]

        if not args.fave:
            f1 = np.delete(f1, np.s_[:64], axis=2)
            f2 = np.delete(f2, np.s_[:128], axis=2)
            f3 = np.delete(f3, np.s_[:256], axis=2)
            f4 = np.delete(f4, np.s_[:512], axis=2)
        if not args.fmax:
            f1 = np.delete(f1, np.s_[-64*2:-64], axis=2)
            f2 = np.delete(f2, np.s_[-128*2:-128], axis=2)
            f3 = np.delete(f3, np.s_[-256*2:-256], axis=2)
            f4 = np.delete(f4, np.s_[-512*2:-512], axis=2)
        if not args.fstd:
            f1 = np.delete(f1, np.s_[-64:], axis=2)
            f2 = np.delete(f2, np.s_[-128:], axis=2)
            f3 = np.delete(f3, np.s_[-256:], axis=2)
            f4 = np.delete(f4, np.s_[-512:], axis=2)

        f = np.zeros((n_vid, n_snp, 1), dtype=np.float32)
        if args.f1:
            f = np.concatenate((f, f1), axis=2)
        if args.f2:
            f = np.concatenate((f, f2), axis=2)
        if args.f3:
            f = np.concatenate((f, f3), axis=2)
        if args.f4:
            f4 = f4.reshape(f4.shape[:-1] + (-1, 4, 128))
            f4 = f4.transpose([0, 1, 3, 2, 4])
            f4 = f4.reshape(f4.shape[:-3] + (-1,))

            f = np.concatenate((f, f4), axis=2)

        f = f[:, :, 1:]
        return f

    def split_data(self, split_base, picked):

        n_data = len(split_base)

        train_idx, eval_idx = [], []
        train_ref, eval_ref = [], []
        for i in range(n_data):
            if split_base[i] in picked:
                eval_idx.append(i)
            else:
                train_idx.append(i)
        for i in np.unique(split_base):
            if i in picked:
                eval_ref.append(i)
            else:
                train_ref.append(i)
        return train_idx, eval_idx, train_ref, eval_ref


class VideoDataset(Dataset):
    r"""
    A Dataset for a folder of videos

    args:
        directory (str): the path to the directory containing all videos
        mode (str, optional): determines whether to read train/test data
    """

    def __init__(self, data):

        self.ref = data['ref']
        self.dis = data['dis']
        self.mos = data['mos']
        # self.framerate = data['fps']
        self.frms = data['frms']
        self.frame_height = data['height']
        self.frame_width = data['width']
        self.n_data = len(data['mos'])

        self.data = data['data']

    def __getitem__(self, index):
        f = self.data[index]
        mos = self.mos[index]

        return (f, mos)

    def __len__(self):
        return self.n_data


class FC_network(nn.Module):
    def __init__(self, dim, hidden=32):
        super(FC_network, self).__init__()

        self.reg = nn.Sequential(
            nn.Linear(dim, hidden * 4),
            nn.ELU(),
            # nn.Dropout(0.5),
            nn.Linear(hidden * 4, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.reg(x)


class VQA_Model_regression(nn.Module):
    def __init__(self, args):
        super(VQA_Model_regression, self).__init__()

        self.regressor = self.init_model(args)

        # init weights
        self._initialize_weights()

    def init_model(self, args):
        dim, dim_list = self.get_dim_info(args)
        model = FC_network(dim=dim)
        return model

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out',
                #                         nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch, snippets, dim = x.shape
        x = x.view((batch * snippets, dim)).contiguous()

        score = self.regressor(x)
        score = torch.mean(score.view(batch, snippets), dim=1)
        return score

    def get_dim_info(self, args):
        dim = []
        scale = 0
        if args.fave:
            scale += 1
        if args.fmax:
            scale += 1
        if args.fstd:
            scale += 1

        if args.f1:
            dim.append(64 * scale)
        if args.f2:
            dim.append(128 * scale)
        if args.f3:
            dim.append(256 * scale)
        if args.f4:
            dim.append(512 * scale)

        dimension = 0
        for d in dim:
            dimension += d
        return dimension, dim


def fn_fix_seed(seed=12318):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('--' * 12)
    print('seed: %d' % seed)


def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            total_param += num_param
    return total_param


def run(args):

    if args.seed > 0:
        fn_fix_seed(args.seed)

    exp_id = args.exp_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predataset = PreDataset(args)

    train_dataset = VideoDataset(predataset.data['train'])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                               shuffle=True, drop_last=True)

    eval_dataset = VideoDataset(predataset.data['test'])
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=eval_dataset.n_data,
                                              shuffle=False, drop_last=False)

    model = VQA_Model_regression(args).to(device)
    print('number of trainable parameters = ', count_parameters(model))

    args.decay_interval = int(args.epoch / 10)
    args.decay_ratio = 0.8

    criterion = nn.SmoothL1Loss()  # smooth L1 loss
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)

    best_val_criterion = -1  # SROCC min

    best_SROCC, best_KROCC, best_PLCC, best_RMSE = 0, 0, 0, 0
    best_pred, best_test = [], []

    for epoch in range(args.epoch):
        # ----------- Train ----------------
        model.train()
        for i, (f_input, mos) in enumerate(train_loader):

            f_input = f_input.to(device).float()
            mos = mos.to(device).float()

            optimizer.zero_grad()

            outputs = model(f_input)
            loss = criterion(outputs.view(mos.shape), mos)

            loss.backward()
            optimizer.step()
        # scheduler.step()

        # ----------- Eval ----------------
        model.eval()
        y_pred, y_label = [], []
        with torch.no_grad():
            for i, (f_input, mos) in enumerate(eval_loader):
                y_label.append(mos.cpu().float().numpy())

                f_input = f_input.to(device).float()
                outputs = model(f_input)

                y_pred.append(outputs.cpu().float().numpy())

        y_label = np.asarray(y_label).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        PLCC = stats.pearsonr(y_pred, y_label)[0]
        SROCC = stats.spearmanr(y_pred, y_label)[0]
        RMSE = np.sqrt(((y_pred - y_label) ** 2).mean())
        KROCC = stats.stats.kendalltau(y_pred, y_label)[0]

        # Update the model with the best val_SROCC
        if SROCC > best_val_criterion:
            print("EXP-{:02d} update best_val_criterion in epoch {:03d}, "
                  "SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(
                exp_id, epoch, SROCC, KROCC, PLCC, RMSE))

            best_val_criterion = SROCC  # update best val SROCC

            best_SROCC = SROCC
            best_KROCC = KROCC
            best_PLCC = PLCC
            best_RMSE = RMSE
            best_epoch = epoch

            best_pred = y_pred
            best_label = y_label

    perf = np.concatenate(
        (np.asarray([best_epoch, best_SROCC, best_KROCC, best_PLCC, best_RMSE]), best_pred, best_label),
        0).reshape((1, -1))

    np.save('{}/{}-{}-EXP-{:02d}.npy'.format(args.save, args.database, args.prefix, args.exp_id), perf)
    print('========' * 10)

    return perf


def get_combinations(database, output=None):
    # This only work in the condition that `n_ref` ranges from 8 to 12
    # That is, 8 * 0.2 -> 2 <- 12 * 0.2
    # for 8/2 splitting

    # the number of reference videos
    n_ref = {'LIVE': 10, 'CSIQ': 12, 'IVPL': 10, 'EPFL-PoliMI': 12, 'IVC-IC': -1, 'LIVE-Mobile': 10}
    if database not in n_ref.keys():
        print('No such database-info found~ Please revise the code and provide the info.')
        raise NotImplementedError

    n_orgs = int(n_ref[database])
    if n_orgs > 0:
        comb = np.asarray(list(combinations(range(n_orgs), 2))).astype('int')
    elif n_orgs == -1:
        comb = []
        for i in range(0, 49, 3):
            comb.append(np.asarray(range(i, i + 12)))
        for i in range(0, 37, 3):
            comb.append(np.asarray(range(i, i + 24, 2)))
        for i in range(2, 25, 2):
            comb.append(np.asarray(range(i+1, i + 36, 3)))
        for i in range(0, 13, 2):
            comb.append(np.asarray(range(i+3, i + 48, 4)))
        comb.append(np.asarray(range(3, 60, 5)))

        comb = np.asarray(comb).astype('int')
    else:
        print('No such database info~ Please revise the code and provide the info.')
        raise NotImplementedError

    print('database: {} - using {} combinations to verify.'.format(database, len(comb)))

    if output is not None and os.path.exists(output):
        np.savetxt('{}/split_index.{}.txt'.format(output, database), comb, fmt='%d')
        
    return comb


def check_opt(args):
    """
        check options if proper
            args.f1 | args.f2 | args.f3 | args.f4
            args.fave | args.fmax | args.fstd
    """
    count_f = args.f1 + args.f2 + args.f3 + args.f4
    count_p = args.fave + args.fmax + args.fstd

    if count_f < 1:
        print('please provide the type of features')
        raise NotImplementedError
    if count_p < 1:
        print('please provide the type of pooling')
        raise NotImplementedError

    print('No.of.features: {} + No.of.pooling: {}.'.format(count_f, count_p))
    print('options check successfully. {}'.format(u'\u2713'))


if __name__ == "__main__":

    # Settings
    # `batch_size` is set according to the number of distorted videos for each reference
    # `learning_rate` is set to 1e-4 generally
    #
    # +----------------------------------------------------------------------+
    # |  DataBase  | LIVE | CSIQ | IVPL | IVC-IC | EPFL-PoliMI | LIVE-Mobile |
    # | -----------+------|------|------|--------|-------------|-------------|
    # | batch_size |  15  |  18  |   4  |   16   |       12    |    16       |
    # | -----------+---------------------------------------------------------|
    # | learning_r |                         1e-4                            |
    # +----------------------------------------------------------------------+

    parser = argparse.ArgumentParser(description='HEKE finetuning for BVQA (Intra-Dataset)')
    parser.add_argument('--base_path', default='/mnt/disk/yongxu_liu/datasets', type=str, help='path to datasets')
    parser.add_argument('--load', default='', type=str,
                        help='path to feature-loading')
    parser.add_argument('--save', default='', type=str,
                        help='path to save')
    parser.add_argument('--prefix', default='', type=str,
                        help='prefix for save')

    parser.add_argument('--epoch', default=300, type=int, help=' training epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=2, type=int, help='train batch size')

    parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 regularization (default: 1e-4)')
    parser.add_argument('--seed', default=12318, type=int, help='seed for reproducibility (-1 means randomness)')

    parser.add_argument('--reverse', action='store_true',
                        help='run combinations reversely')
    parser.add_argument('--database', default='LIVE', type=str,
                        help='database name (default: LIVE)')

    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val splits (default: 0)')
    parser.add_argument('--mos', action='store_true',
                        help='the groundtruth is MOS (using DMOS = 10 - MOS) | suit for EPFL-PoliMI & IVC-IC')

    parser.add_argument('--f1', action='store_true', help='use f1 features')
    parser.add_argument('--f2', action='store_true', help='use f2 features')
    parser.add_argument('--f3', action='store_true', help='use f3 features')
    parser.add_argument('--f4', action='store_true', help='use f4 features')

    parser.add_argument('--fave', action='store_true', help='use average features')
    parser.add_argument('--fmax', action='store_true', help='use max features')
    parser.add_argument('--fstd', action='store_true', help='use std features')

    args = parser.parse_args()

    try:
        os.environ['IPYTHONENABLE']

        print('in DEBUG mode')
        os.environ["CUDA_VISIBLE_DEVICES"] = '2'
        args.save = './outputs_intra'
        args.database = 'IVPL'
        args.load = './data/r2p1d_HEKE_4/{}.feat_nf-12.pkl'.format(args.database)

        args.epoch = 10

        args.f1 = args.f2 = args.f3 = args.f4 = True
        args.fave = args.fmax = args.fstd = True

        args.batch_size = 4

    except KeyError:
        print('in Release mode')

    args.info_file = '{}/{}/{}_list_for_VQA.txt'.format(args.base_path, args.database, args.database)

    print(args)
    check_opt(args)

    if not os.path.exists(args.load):
        print('No such feature data found~ Please extract features first.')
        raise NotImplementedError

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    args.save = '{}/{}'.format(args.save, args.database)
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    comb = get_combinations(database=args.database, output=args.save)

    srccs, perfs = [], []
    exp_id = args.exp_id
    for i in range(exp_id, len(comb)):
        if args.reverse:
            print('running reversely')
            index = len(comb) - 1 - i
        else:
            index = i

        target = comb[index]
        args.target = target

        perf = run(args)

        srccs.append(perf[0][1])
        perfs.append(perf)
        args.exp_id += 1

        #
    srccs = np.array(srccs)
    med = np.median(srccs)
    meanv = np.mean(srccs)
    percent = np.percentile(srccs, [25, 50, 75])
    print('\n------------------')
    print('performance: med-srcc: {:.4f}, mean-srcc: {:.4f}'.format(med, meanv))
    print('Quartiles: %.4f - %.4f - %.4f ' % (percent[0], percent[1], percent[2]))
    print('range: [%.4f - %.4f]' % (srccs.min(), srccs.max()))

    with open('{}/{}-{}-ALL.pkl'.format(args.save, args.database, args.prefix), 'wb') as f:
        pickle.dump(perfs, f)





# ==========================================================================================


