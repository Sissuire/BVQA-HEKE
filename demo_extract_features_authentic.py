import os
import cv2
import random
import numpy as np
import argparse
import timeit
import pickle
import torch
import torch.nn as nn
from scipy import stats
from init_models.resnet2p1d import generate_model
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tf
from math import ceil

from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class DatasetValidation_with_Average(Dataset):
    r"""
    A Validation Dataset for a folder of videos

    """

    def __init__(self, dataset='LIVE', basep='/mnt/diskplus/datasets/images4database_downscale', nsnippet=10):

        self.basep = basep
        self.dataset = dataset

        # pre-load the test data for fast evaluation without data preparing
        self.nfrm = 8
        self.nsnippet = nsnippet
        self.n_data, self.info = self.get_test_info(dataset)
        # self.__getitem__(index=10)

        # self.__getitem__(28)

    def get_test_batch(self, file):

        cap = cv2.VideoCapture(file)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frms = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if min(width, height) > 760:
            scale = ceil(min(width, height) / 256) - 1
        else:
            scale = ceil(min(width, height) / 256)

        scaleW = width // scale
        scaleH = height // scale
        transform = tf.Compose([
            # tf.Resize([scaleH, scaleW]),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        ts = 2 if frms < 480 else 4
        N = self.nfrm
        T = self.nsnippet
        offset = 10

        stride = (frms - offset * 2 - ts * N) / (T - 1)
        pos = np.asarray(range(T)) * stride + offset - 1
        pos = pos.astype('int')

        vid = np.zeros((T, 3, N, scaleH, scaleW), dtype=np.float32)

        # buffer = skvideo.io.vread(file)
        buffer = []
        ret, frame = cap.read()
        while ret:
            buffer.append(frame)
            ret, frame = cap.read()
        for i in range(T):
            thispos = pos[i]
            for j in range(N):
                no_id = thispos + ts * j

                img = buffer[no_id].astype('float32')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.AveragePooling(img.astype('float32'), scale) / 255.
                img = transform(img)
                vid[i, :, j] = np.copy(img)

        return vid

    def get_test_info(self, dataset):
        info_file = '{}/{}/{}_info.txt'.format(self.basep, dataset, dataset)
        dis_list, score_list = [], []
        with open(info_file, 'r') as f:
            for line in f:
                dis, score = line.split()

                dis_list.append('{}/{}/{}'.format(self.basep, dataset, dis))
                score_list.append(float(score))

        dis_list = np.asarray(dis_list)
        score_list = np.array(score_list, dtype='float32')

        score_list = 1 + 8.5 * (score_list - score_list.min()) / (score_list.max() - score_list.min())

        return len(score_list), {'dis': dis_list,
                                 'mos': score_list}

    def AveragePooling(self, img, kernel=2):

        # result = block_reduce(img, block_size=(kernel, kernel, 1), func=np.mean)
        H, W, C = img.shape
        Hc = H // kernel
        Wc = W // kernel
        H = Hc * kernel
        W = Wc * kernel

        # crop if possible
        img = img[:H, :W]

        # integral
        data = np.reshape(img, [H * Wc, kernel, C])
        data = np.sum(data, axis=1)

        data = np.reshape(np.transpose(np.reshape(data, [H, Wc, C]), [1, 0, 2]), [Hc * Wc, kernel, C])
        data = np.sum(data, axis=1)

        result = np.transpose(np.reshape(data, [Wc, Hc, C]), [1, 0, 2]) / (kernel * kernel)
        return result

    def __getitem__(self, index):
        file = self.info['dis'][index]

        dis = self.get_test_batch(file)
        mos = self.info['mos'][index]

        return dis, mos

    def __len__(self):
        return self.n_data


class HEKE_BVQA_r2p1d(nn.Module):
    def __init__(self, depth=18, group=1, hidden=32):
        """Multi-Knowledge Ensemble Learning for VQA.
        - MS-SSIM
        - GMSD
        - ST-GMSD
        - ST-RRED

        Args:
            depth (int): resnet depth, default to 18
            group (int): group convolution for each method
            hidden (int): hidden size for FC layer, default to 32
        """
        super(HEKE_BVQA_r2p1d, self).__init__()
        group = group * 4
        self.group = group
        self.resnet = generate_model(model_depth=depth)  # output [512 x M x N]

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # feature distribution is varied among contents. How to solve?
        self.maxpool = nn.AdaptiveMaxPool3d((1, 1, 1))

        self.group_conv_1 = nn.Conv2d(in_channels=512, out_channels=hidden*group*4,
                                      kernel_size=(2, 1),
                                      stride=1,
                                      padding=0,
                                      groups=group)
        self.group_conv_2 = nn.Conv2d(in_channels=hidden*group*4, out_channels=hidden*group,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=group)
        self.group_conv_3 = nn.Conv2d(in_channels=hidden*group, out_channels=group,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      groups=group)
        self.act = nn.ELU()

        # init weights
        self._initialize_weights()

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
        f1, f2, f3, f4 = self.resnet(x)

        x = f4
        x1 = self.maxpool(x)
        x2 = self.avgpool(x)

        x = torch.cat((x1, x2), 2).flatten(3)
        x = self.group_conv_1(x)
        x = self.act(x)
        x = self.group_conv_2(x)
        x = self.act(x)
        x = self.group_conv_3(x)

        # ---------------------
        f1_ave = self.avgpool(f1).flatten(1)
        f1_max = self.maxpool(f1).flatten(1)
        f1_std = torch.std(f1, dim=(2, 3, 4), keepdim=False)

        f2_ave = self.avgpool(f2).flatten(1)
        f2_max = self.maxpool(f2).flatten(1)
        f2_std = torch.std(f2, dim=(2, 3, 4), keepdim=False)

        f3_ave = self.avgpool(f3).flatten(1)
        f3_max = self.maxpool(f3).flatten(1)
        f3_std = torch.std(f3, dim=(2, 3, 4), keepdim=False)

        f4_ave = self.avgpool(f4).flatten(1)
        f4_max = self.maxpool(f4).flatten(1)
        f4_std = torch.std(f4, dim=(2, 3, 4), keepdim=False)
        return x.flatten(1), torch.cat([f1_ave, f1_max, f1_std, f2_ave, f2_max, f2_std,
                                        f3_ave, f3_max, f3_std, f4_ave, f4_max, f4_std], dim=1)


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
            if param.dim() > 1:
                print(name, ': ', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ': ', num_param)
            total_param += num_param
    return total_param


def run_features(args):
    # if args.seed > 0:
    #     fn_fix_seed(args.seed)

        # [DEBUG MODE]
    try:
        os.environ['IPYTHONENABLE']

        print('in DEBUG mode')
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'
        # args.load = './pretrained/r2p1d_model_HEKE4.pkl'
    except KeyError:
        print('in Release mode')

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    database = args.database

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testdataset = DatasetValidation_with_Average(dataset=database, nsnippet=10,
                                                 basep='/mnt/disk/yongxu_liu/datasets/in_the_wild')
    test_loader = DataLoaderX(testdataset, batch_size=1, shuffle=False)

    model = HEKE_BVQA_r2p1d()
    state_dict = torch.load(args.load)
    state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)

    for p in model.parameters():
        p.requires_grad = False

    if torch.cuda.device_count() > 1 and args.multi_gpu is True:
        device_ids = range(0, torch.cuda.device_count())
        model = torch.nn.DataParallel(model.to(device), device_ids=device_ids)
        print("muti-gpu mode enabled, use {0:d} gpus".format(torch.cuda.device_count()))

    else:
        model = model.to(device)
        print('use {0}'.format('cuda' if torch.cuda.is_available() else 'cpu'))
    print('number of trainable parameters = ', count_parameters(model))

    print(database)
    print(args.model)

    # ---------------------------------
    model.eval()
    feat = []
    pred_gmsd, pred_stgmsd, pred_strred, pred_msssim, y_label = [], [], [], [], []
    with torch.no_grad():
        for i, (vid, mos) in enumerate(test_loader):
            print('{} '.format(i), end='', flush=True)
            batch = vid.shape[0]
            ns = vid.shape[1]
            sh2 = vid.shape[2:]

            vid = vid.view(torch.Size([ns, batch]) + sh2).to(device)
            y_label.append(mos.cpu().numpy())

            # vid = vid.to(device).float()
            # mos = mos.to(device).float()

            ff = np.zeros((10, 2880), dtype=np.float32)
            outputs = torch.zeros((1, 4), dtype=torch.float)
            for iii, vvi in enumerate(vid):
                outputi, ffi = model(vvi)
                ff[iii, :] = ffi.cpu().numpy()
                outputs += outputi.cpu()

            gmsd, stgmsd, strred, msssim = torch.mean(outputs, 0).numpy() / 10

            pred_gmsd.append(gmsd)
            pred_stgmsd.append(stgmsd)
            pred_strred.append(strred)
            pred_msssim.append(msssim)

            feat.append(ff)

    y_label = np.asarray(y_label).reshape(-1)
    pred_gmsd = np.asarray(pred_gmsd).reshape(-1)
    pred_stgmsd = np.asarray(pred_stgmsd).reshape(-1)
    pred_strred = np.asarray(pred_strred).reshape(-1)
    pred_msssim = np.asarray(pred_msssim).reshape(-1)

    SROCC_gmsd = stats.spearmanr(pred_gmsd, y_label)[0]
    SROCC_stgmsd = stats.spearmanr(pred_stgmsd, y_label)[0]
    SROCC_strred = stats.spearmanr(pred_strred, y_label)[0]
    SROCC_msssim = stats.spearmanr(pred_msssim, y_label)[0]

    print('---' * 7)
    print('{} = {}'.format('DMOS', y_label))
    print('{} = {}'.format('gmsd', pred_gmsd))
    print('{} = {}'.format('stgmsd', pred_stgmsd))
    print('{} = {}'.format('strred', pred_strred))
    print('{} = {}'.format('msssim', pred_msssim))
    print('---' * 7)
    print('')

    print(' ========= {} ========='.format(args.database))
    print('')
    print('  gmsd: {:.4f}'.format(SROCC_gmsd))
    print('stgmsd: {:.4f}'.format(SROCC_stgmsd))
    print('strred: {:.4f}'.format(SROCC_strred))
    print('msssim: {:.4f}'.format(SROCC_msssim))
    print('-----' * 5)

    with open('{p}/{db}.feat_{pf}.pkl'.format(p=args.save, db=args.database, pf=args.prefix), 'wb') as f:
        pickle.dump({'features': feat, 'mos': y_label}, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Heterogeneous Knowledge Ensemble for NR-VQA')
    parser.add_argument('--model_depth', default=18, type=int,
                        help='model: ResNet{10|18|34|50}. Default to 18')  # we use ResNet18 in the work
    parser.add_argument('--model', default='baseline_v3', type=str,
                        help='which model name to run and save')
    parser.add_argument('--load', default='./pretrained/r2p1d_model_HEKE4.pkl', type=str,
                        help='full name of model to load')
    parser.add_argument('--save', default='./data/r2p1d_HEKE_4',
                        help='path to save features')
    parser.add_argument('--prefix', default='',
                        help='prefix for saving')
    parser.add_argument('--database', default='KoNViD', type=str,
                        help='database name (default: KoNViD)')

    args = parser.parse_args()
    print(args)

    run_features(args=args)

# ==========================================================================================


